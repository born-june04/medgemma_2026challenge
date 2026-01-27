
from __future__ import annotations

import math
from collections import Counter, defaultdict
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, BatchSampler, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable

# Uses the small head you added earlier
from .audio_head import AudioClassifier


# ----------------------------
# Dataset discovery: <root>/<class>/CSI/*.png
# ----------------------------
class AudioImageCSIDataset(Dataset):
    """
    Discovers class-labeled PNG scalograms under:
      dataset_root/
        <class_name>/
          CSI/
            *.png

    Label = name of the folder directly above 'CSI'.
    """

    def __init__(self, dataset_root: str, recursive: bool = True) -> None:
        self.root = Path(dataset_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # Find every 'CSI' directory
        csi_dirs: List[Path] = (
            [p for p in self.root.rglob("CSI") if p.is_dir()]
            if recursive
            else [p for p in (self.root.glob("*/CSI")) if p.is_dir()]
        )

        class_dirs = [d.parent for d in csi_dirs]
        classes = sorted({d.name for d in class_dirs})
        if not classes:
            raise RuntimeError(
                f"No class folders with 'CSI' subdirectories found under {self.root}."
            )

        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}

        samples: List[Tuple[str, int]] = []
        for class_dir in class_dirs:
            cls = class_dir.name
            cidx = self.class_to_idx[cls]
            csi_dir = class_dir / "CSI"
            files = list(csi_dir.rglob("*.png")) if recursive else list(csi_dir.glob("*.png"))
            for f in files:
                if f.is_file():
                    samples.append((str(f), cidx))

        if not samples:
            raise RuntimeError(
                f"No PNG files found in any 'CSI' subfolders under {self.root}."
            )

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


# ----------------------------
# Embedding precomputation (encoder-agnostic)
# ----------------------------
@torch.no_grad()
def precompute_embeddings(encoder, dataset: Dataset, out_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs a frozen encoder over the dataset and returns a compact cache.

    Encoder contract:
      - encoder.encode(path) -> 1D torch.Tensor
      - encoder.metadata.embedding_dim: int
      - (optional) encoder._load_model(): will be called if present

    Returns a dict with:
      - embeddings: FloatTensor [N, D]
      - labels: LongTensor [N]
      - class_to_idx: Dict[str, int]
      - idx_to_class: Dict[int, str]
      - embedding_dim: int
    """
    # Ensure underlying model is ready (if the encoder uses lazy loading)
    if hasattr(encoder, "_load_model"):
        encoder._load_model()

    D = int(encoder.metadata.embedding_dim)
    N = len(dataset)

    embs = torch.empty((N, D), dtype=torch.float32)
    labels = torch.empty((N,), dtype=torch.long)

    for i in range(N):
        path, y = dataset[i]
        e = encoder.encode(path)
        if isinstance(e, torch.Tensor):
            e = e.detach().cpu()
        if e.ndim != 1 or e.shape[0] != D:
            raise ValueError(f"Unexpected embedding shape {tuple(e.shape)}; expected ({D},)")
        embs[i] = e
        labels[i] = y

    pack: Dict[str, Any] = {
        "embeddings": embs,
        "labels": labels,
        "class_to_idx": getattr(dataset, "class_to_idx", {}),
        "idx_to_class": getattr(dataset, "idx_to_class", {}),
        "embedding_dim": D,
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(pack, out_path)

    return pack


# ----------------------------
# Head training loop (≤ 100 LOC)
# ----------------------------
class BalancedBatchSampler(BatchSampler):
    """Yield class-balanced batches with equal samples per class."""

    def __init__(self, labels: torch.Tensor, batch_size: int, seed: int = 1337) -> None:
        self.labels = labels.detach().cpu().tolist()
        self.num_classes = len(set(self.labels))
        if batch_size % self.num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of num_classes ({self.num_classes})"
            )
        self.batch_size = batch_size
        self.per_class = batch_size // self.num_classes
        self.rng = random.Random(seed)
        self.num_batches = len(self.labels) // self.batch_size

        self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[int(label)].append(idx)
        for cls, idxs in self.class_to_indices.items():
            if not idxs:
                raise ValueError(f"No samples found for class {cls}")

    def __iter__(self):
        # Oversample with replacement to ensure equal samples per class per batch
        for _ in range(self.num_batches):
            batch: List[int] = []
            for cls in sorted(self.class_to_indices.keys()):
                indices = self.class_to_indices[cls]
                batch.extend(self.rng.choices(indices, k=self.per_class))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def _binary_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Compute ROC AUC for binary labels using trapezoidal rule."""
    y_true = y_true.float()
    pos = y_true.sum().item()
    neg = y_true.numel() - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]
    tps = torch.cumsum(y_true, dim=0)
    fps = torch.cumsum(1 - y_true, dim=0)
    tpr = tps / pos
    fpr = fps / neg
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])
    return torch.trapz(tpr, fpr).item()


def _compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute accuracy, macro ROC AUC, sensitivity, and specificity."""
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    num_classes = probs.shape[1]

    acc = (preds == labels).float().mean().item()

    # Macro ROC AUC (one-vs-rest)
    aucs: List[float] = []
    for cls in range(num_classes):
        y_true = (labels == cls).int()
        auc = _binary_auc(y_true, probs[:, cls])
        if not (auc != auc):  # not NaN
            aucs.append(auc)
    auc_roc = float(sum(aucs) / len(aucs)) if aucs else float("nan")

    # Sensitivity / Specificity (macro)
    sensitivities: List[float] = []
    specificities: List[float] = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        tn = ((preds != cls) & (labels != cls)).sum().item()
        if tp + fn > 0:
            sensitivities.append(tp / (tp + fn))
        if tn + fp > 0:
            specificities.append(tn / (tn + fp))
    sensitivity = float(sum(sensitivities) / len(sensitivities)) if sensitivities else float("nan")
    specificity = float(sum(specificities) / len(specificities)) if specificities else float("nan")

    return {
        "acc": acc,
        "auc_roc": auc_roc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

def train_audio_classifier_head(
    emb_pack: Dict[str, Any],
    ckpt_path: str,
    *,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 30,
    val_ratio: float = 0.2,
    early_stopping_patience: int = 5,
    amp: bool = True,
    balanced_sampling: bool = False,
) -> Dict[str, Any]:
    """
    Trains a small classifier head on cached embeddings with early stopping.
    Saves the best checkpoint to `ckpt_path` and returns the loaded best checkpoint dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X: torch.Tensor = emb_pack["embeddings"]  # [N, D]
    y: torch.Tensor = emb_pack["labels"]      # [N]
    D = int(emb_pack["embedding_dim"])
    class_to_idx: Dict[str, int] = emb_pack.get("class_to_idx", {})
    num_classes = len(class_to_idx) if class_to_idx else int(y.max().item()) + 1

    # Split train/val
    N = X.shape[0]
    # val_size = max(1, int(math.floor(N * val_ratio)))
    val_size = 0
    train_size = N - val_size
    g = torch.Generator().manual_seed(1337)
    train_ds, val_ds = random_split(TensorDataset(X, y), [train_size, val_size], generator=g)
    if balanced_sampling:
        train_labels = y[train_ds.indices]
        batch_sampler = BalancedBatchSampler(train_labels, batch_size=batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    val_loader = None

    # Model
    model = AudioClassifier(input_dim=D, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Class weights for imbalance
    counts = Counter(y.tolist())
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    best_val = float("inf")
    epochs_no_improve = 0
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    epoch_pbar = tqdm(range(1, max_epochs + 1), desc="Epochs")
    for epoch in epoch_pbar:
        # ---- Train ----
        model.train()
        running_loss = 0.0
        seen = 0
        train_logits = []
        train_labels = []
        train_pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for xb, yb in train_pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * yb.size(0)
            seen += yb.size(0)
            train_logits.append(logits.detach().cpu())
            train_labels.append(yb.detach().cpu())

        train_loss = running_loss / max(1, seen)
        train_metrics = _compute_metrics(torch.cat(train_logits), torch.cat(train_labels))

        # ---- Validate ----
        model.eval()
        val_losses = 0.0
        v_seen = 0
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                v_loss = nn.functional.cross_entropy(logits, yb)
                val_losses += v_loss.item() * yb.size(0)
                v_seen += yb.size(0)
                val_logits.append(logits.detach().cpu())
                val_labels.append(yb.detach().cpu())

        val_loss = val_losses / max(1, v_seen)
        val_metrics = _compute_metrics(torch.cat(val_logits), torch.cat(val_labels))
        scheduler.step(val_loss)

        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_metrics['acc']:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_metrics['acc']:.4f}",
            val_auc=f"{val_metrics['auc_roc']:.4f}",
            val_sens=f"{val_metrics['sensitivity']:.4f}",
            val_spec=f"{val_metrics['specificity']:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        # Early stopping on best val_loss
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": D,
                    "num_classes": num_classes,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "class_to_idx": class_to_idx,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

    # Return the best checkpoint dict for reporting
    best = torch.load(ckpt_path, map_location="cpu")
    return best


__all__ = [
    "AudioImageCSIDataset",
    "precompute_embeddings",
    "train_audio_classifier_head",
]

#!/usr/bin/env python3
"""
Train a small classifier head on top of frozen image embeddings (CXR).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Ensure project root is on path when running as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from huggingface_hub import login
from Agent2_Image.encoders import get_image_encoder
from Agent2_Image.classifiers.image_head import ImageClassifier
from Agent2_Image.classifiers.image_training import (
    ImageCXRDataset,
    _compute_metrics,
    finetune_attention_encoder,
    precompute_embeddings,
    train_image_classifier_head,
)
from Agent2_Image.utils.gradcam_utils import (
    save_overlay,
    save_overlay_with_colorbar,
    save_raw_heatmap,
    save_original,
)


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _val_split_loaders(
    emb_pack: Dict[str, Any], batch_size: int = 256, val_ratio: float = 0.2, seed: int = 1337
) -> Tuple[DataLoader, DataLoader]:
    X: torch.Tensor = emb_pack["embeddings"]  # [N, D]
    y: torch.Tensor = emb_pack["labels"]      # [N]
    N = X.shape[0]
    # val_size = max(1, int(N * val_ratio))
    val_size = 0
    train_size = N - val_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(TensorDataset(X, y), [train_size, val_size], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, train_loader


def _evaluate_on_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, Dict[str, float]]:
    ce = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    losses = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            losses += loss.item() * yb.size(0)
            total += yb.numel()
            all_logits.append(logits.detach().cpu())
            all_labels.append(yb.detach().cpu())
    val_loss = losses / max(1, total)
    metrics = _compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    return val_loss, metrics


def _load_head_from_ckpt_dict(ckpt: Dict[str, Any]) -> Tuple[ImageClassifier, Dict[int, str]]:
    input_dim = int(ckpt["input_dim"])
    num_classes = int(ckpt["num_classes"])
    hidden_dim = ckpt.get("hidden_dim", None)
    dropout = float(ckpt.get("dropout", 0.1))

    model = ImageClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    c2i = ckpt.get("class_to_idx", {})
    if c2i:
        idx_to_class = {v: k for k, v in c2i.items()}
    else:
        idx_to_class = {i: str(i) for i in range(num_classes)}

    return model, idx_to_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image classifier head on cached embeddings")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--force-recache", action="store_true", help="Ignore existing cache and recompute")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("hf_token"):
        os.environ["HF_TOKEN"] = str(cfg["hf_token"])
        login(token=os.environ["HF_TOKEN"])

    icfg = (cfg.get("image_classifier") or {})
    dataset_root = icfg.get("dataset_root")
    cache_path = icfg.get("cache_path", "artifacts/image_embeddings_cache.pt")
    ckpt_path = icfg.get("checkpoint_path", "artifacts/image_classifier_head.pt")
    ft_cfg = (cfg.get("image_finetune") or {})

    if not dataset_root:
        raise ValueError("configs.config.yaml -> image_classifier.dataset_root must be set")
    dataset_root = str(dataset_root)

    use_stub = bool(cfg.get("use_stub_encoders", True))
    image_checkpoint = cfg.get("image_checkpoint")
    encoder = get_image_encoder(use_stub=use_stub, checkpoint_path=image_checkpoint)

    ds = ImageCXRDataset(dataset_root)
    print(f"[dataset] Found {len(ds)} files across {len(ds.idx_to_class)} classes:")
    print(json.dumps(ds.idx_to_class, indent=2))

    if ft_cfg.get("enabled", False):
        print("[finetune] Fine-tuning attention weights on CXR data...")
        finetune_attention_encoder(
            encoder,
            ds,
            batch_size=int(ft_cfg.get("batch_size", 8)),
            lr_attn=float(ft_cfg.get("lr_attn", 1e-6)),
            lr_head=float(ft_cfg.get("lr_head", 1e-4)),
            weight_decay=float(ft_cfg.get("weight_decay", 1e-4)),
            max_epochs=int(ft_cfg.get("max_epochs", 3)),
            val_ratio=float(ft_cfg.get("val_ratio", 0.2)),
            amp=bool(ft_cfg.get("amp", True)),
            balanced_sampling=bool(ft_cfg.get("balanced_sampling", True)),
        )
        print("[finetune] Done.")

    cache_path = str(cache_path)
    if (not args.force_recache) and Path(cache_path).exists():
        emb_pack = torch.load(cache_path, map_location="cpu")
        print(f"[cache] Loaded cached embeddings: {emb_pack['embeddings'].shape} from {cache_path}")
    else:
        emb_pack = precompute_embeddings(encoder, ds, out_path=cache_path)
        print(f"[cache] Saved embeddings to {cache_path}: {emb_pack['embeddings'].shape}")

    hidden_dim = icfg.get("hidden_dim", None)
    dropout = float(icfg.get("dropout", 0.1))
    batch_size = int(icfg.get("batch_size", 64))
    lr = float(icfg.get("lr", 1e-3))
    weight_decay = float(icfg.get("weight_decay", 1e-4))
    max_epochs = int(icfg.get("max_epochs", 30))
    val_ratio = float(icfg.get("val_ratio", 0.2))
    early_stopping_patience = int(icfg.get("early_stopping_patience", 5))
    amp = bool(icfg.get("amp", True))
    balanced_sampling = bool(icfg.get("balanced_sampling", True))

    best_ckpt = train_image_classifier_head(
        emb_pack=emb_pack,
        ckpt_path=ckpt_path,
        hidden_dim=hidden_dim,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        val_ratio=val_ratio,
        early_stopping_patience=early_stopping_patience,
        amp=amp,
        balanced_sampling=balanced_sampling,
    )
    print(f"[train] Saved best head checkpoint to: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = _val_split_loaders(emb_pack, batch_size=256, val_ratio=val_ratio, seed=1337)

    model, idx_to_class = _load_head_from_ckpt_dict(best_ckpt)
    model.to(device)

    val_loss, metrics = _evaluate_on_loader(model, val_loader, device)
    print(
        "[eval] "
        f"val_loss={val_loss:.4f}  "
        f"val_acc={metrics['acc']:.4f}  "
        f"auc_roc={metrics['auc_roc']:.4f}  "
        f"sensitivity={metrics['sensitivity']:.4f}  "
        f"specificity={metrics['specificity']:.4f}"
    )
    print("[classes]", json.dumps(idx_to_class, indent=2))

    # After training, generate one Grad-CAM per label (one sample each)
    gcfg = cfg.get("gradcam") or {}
    if gcfg.get("enabled", False):
        output_dir = gcfg.get("output_dir", "outputs/gradcam")
        alpha = float(gcfg.get("alpha", 0.4))
        targets = gcfg.get("targets", ["embedding"])
        label_texts = [ds.idx_to_class[i] for i in range(len(ds.idx_to_class))]

        seen = set()
        for path, label_idx in ds.samples:
            if label_idx in seen:
                continue
            seen.add(label_idx)
            stem = Path(path).stem
            label_name = ds.idx_to_class.get(label_idx, str(label_idx))
            out_dir = Path(output_dir) / stem / label_name

            save_original(path, str(out_dir / "original.png"))

            if "embedding" in targets:
                heatmap = encoder.gradcam(path, target="embedding")
                out_overlay = out_dir / "embedding_overlay.png"
                out_raw = out_dir / "embedding_raw.png"
                save_overlay(path, heatmap, str(out_overlay), alpha=alpha)
                save_raw_heatmap(heatmap, str(out_raw))

            if "classifier" in targets:
                heatmap = encoder.gradcam(path, target="classifier", classifier=model)
                out_overlay = out_dir / "classifier_overlay.png"
                out_raw = out_dir / "classifier_raw.png"
                save_overlay(path, heatmap, str(out_overlay), alpha=alpha)
                save_raw_heatmap(heatmap, str(out_raw))

            if "text" in targets:
                heatmap = encoder.gradcam(
                    path, target="text", texts=label_texts, class_index=label_idx
                )
                out_overlay = out_dir / "text_overlay.png"
                out_raw = out_dir / "text_raw.png"
                save_overlay(path, heatmap, str(out_overlay), alpha=alpha)
                save_raw_heatmap(heatmap, str(out_raw))

            if "attention" in targets:
                heatmap = encoder.attention_rollout(path)
                out_overlay = out_dir / "attention_overlay.png"
                out_raw = out_dir / "attention_raw.png"
                save_overlay(path, heatmap, str(out_overlay), alpha=alpha)
                save_raw_heatmap(heatmap, str(out_raw))

            if "attn_grad" in targets:
                heatmap = encoder.attn_gradcam(
                    path, target="text", texts=label_texts, class_index=label_idx
                )
                out_overlay = out_dir / "attn_grad_overlay.png"
                out_raw = out_dir / "attn_grad_raw.png"
                save_overlay(path, heatmap, str(out_overlay), alpha=alpha)
                save_raw_heatmap(heatmap, str(out_raw))

            if "occlusion" in targets:
                # Mean-baseline occlusion maps (overlay only; no raw files)
                common = dict(
                    texts=label_texts,
                    class_index=label_idx,
                    patch_size=32,
                    stride=32,
                    batch_size=16,
                    baseline="mean",
                )

                heat_dec = encoder.occlusion_map(path, mode="decrease", **common)
                save_overlay_with_colorbar(
                    path,
                    heat_dec,
                    str(out_dir / "occlusion_decrease_overlay.png"),
                    alpha=alpha,
                    title="occlusion decrease = evidence-for",
                )

                heat_inc = encoder.occlusion_map(path, mode="increase", **common)
                save_overlay_with_colorbar(
                    path,
                    heat_inc,
                    str(out_dir / "occlusion_increase_overlay.png"),
                    alpha=alpha,
                    title="occlusion increase = distractor",
                )

            if len(seen) >= len(ds.idx_to_class):
                break


if __name__ == "__main__":
    main()


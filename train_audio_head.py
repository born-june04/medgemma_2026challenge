
#!/usr/bin/env python3
"""
Train a small classifier head on top of frozen audio embeddings.

Usage:
  python train_audio_head.py --config configs/config.yaml

This will:
  - Discover dataset_root/<class>/CSI/*.png
  - Precompute embeddings (or load cached)
  - Train a head with early stopping
  - Save artifacts:
      * artifacts/audio_embeddings_cache.pt
      * artifacts/audio_classifier_head.pt
  - Print class map and validation metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Project imports
from encoders import get_audio_encoder
from classifiers.audio_head import load_head_from_ckpt
from classifiers.audio_training import (
    AudioImageCSIDataset,
    precompute_embeddings,
    train_audio_classifier_head,
)

# ----------------------------
# Helpers
# ----------------------------
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
    val_size = max(1, int(N * val_ratio))
    train_size = N - val_size
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(TensorDataset(X, y), [train_size, val_size], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _evaluate_on_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    ce = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    losses = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            losses += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    val_loss = losses / max(1, total)
    val_acc = correct / max(1, total)
    return val_loss, val_acc


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train audio classifier head on cached embeddings")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--force-recache", action="store_true", help="Ignore existing cache and recompute")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    acfg = (cfg.get("audio_classifier") or {})
    dataset_root = acfg.get("dataset_root")
    cache_path = acfg.get("cache_path", "artifacts/audio_embeddings_cache.pt")
    ckpt_path = acfg.get("checkpoint_path", "artifacts/audio_classifier_head.pt")

    # Basic validations
    if not dataset_root:
        raise ValueError("configs.config.yaml -> audio_classifier.dataset_root must be set")
    dataset_root = str(dataset_root)

    # Instantiate encoder (frozen)
    use_stub = bool(cfg.get("use_stub_encoders", True))
    audio_checkpoint = cfg.get("audio_checkpoint")
    encoder = get_audio_encoder(use_stub=use_stub, checkpoint_path=audio_checkpoint)

    # Build dataset
    ds = AudioImageCSIDataset(dataset_root)
    print(f"[dataset] Found {len(ds)} files across {len(ds.idx_to_class)} classes:")
    print(json.dumps(ds.idx_to_class, indent=2))

    # Precompute embeddings (or load cache)
    cache_path = str(cache_path)
    if (not args.force_recache) and Path(cache_path).exists():
        emb_pack = torch.load(cache_path, map_location="cpu")
        print(f"[cache] Loaded cached embeddings: {emb_pack['embeddings'].shape} from {cache_path}")
    else:
        emb_pack = precompute_embeddings(encoder, ds, out_path=cache_path)
        print(f"[cache] Saved embeddings to {cache_path}: {emb_pack['embeddings'].shape}")

    # Train head
    hidden_dim = acfg.get("hidden_dim", None)  # None => Linear; int => small MLP
    dropout = float(acfg.get("dropout", 0.1))
    batch_size = int(acfg.get("batch_size", 64))
    lr = float(acfg.get("lr", 1e-3))
    weight_decay = float(acfg.get("weight_decay", 1e-4))
    max_epochs = int(acfg.get("max_epochs", 30))
    val_ratio = float(acfg.get("val_ratio", 0.2))
    early_stopping_patience = int(acfg.get("early_stopping_patience", 5))
    amp = bool(acfg.get("amp", True))

    best_ckpt = train_audio_classifier_head(
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
    )
    print(f"[train] Saved best head checkpoint to: {ckpt_path}")

    # Report validation metrics using the same split convention (seed=1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = _val_split_loaders(emb_pack, batch_size=256, val_ratio=val_ratio, seed=1337)

    model, idx_to_class = load_head_from_ckpt(ckpt_path)
    model.to(device)

    val_loss, val_acc = _evaluate_on_loader(model, val_loader, device)
    print(f"[eval] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    print("[classes]", json.dumps(idx_to_class, indent=2))


if __name__ == "__main__":
    main()

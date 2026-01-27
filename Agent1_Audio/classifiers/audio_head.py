from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    """
    Minimal classifier head to sit on top of a frozen audio embedding.

    - If hidden_dim is None: LayerNorm -> Dropout -> Linear  (logistic regression)
    - If hidden_dim is int:  LayerNorm -> Dropout -> Linear(input->hidden) -> GELU
                             -> Dropout -> Linear(hidden->num_classes)  (small MLP)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            self.net = nn.Sequential(
                # nn.LayerNorm(input_dim),
                # nn.Dropout(p=dropout),
                nn.Linear(input_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, input_dim] or [input_dim]
        Returns:
            logits: Tensor of shape [batch, num_classes]
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def create_audio_classifier(
    input_dim: int,
    num_classes: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
) -> AudioClassifier:
    """
    Convenience constructor to keep callsites clean.
    """
    return AudioClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def load_head_from_ckpt(ckpt_path: str) -> Tuple[AudioClassifier, Dict[int, str]]:
    """
    Load a saved classifier head checkpoint.

    Expects a checkpoint dict with keys:
      - "model_state": state_dict of the head
      - "input_dim": int
      - "num_classes": int
      - "hidden_dim": Optional[int]
      - "dropout": float
      - "class_to_idx": Dict[str, int]  (optional but recommended)

    Returns:
      model: AudioClassifier in eval() mode
      idx_to_class: Dict[int, str] mapping class index -> class name
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    input_dim = int(ckpt["input_dim"])
    num_classes = int(ckpt["num_classes"])
    hidden_dim = ckpt.get("hidden_dim", None)
    dropout = float(ckpt.get("dropout", 0.1))

    model = AudioClassifier(
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
        # Fallback to numeric string labels if class names weren't saved
        idx_to_class = {i: str(i) for i in range(num_classes)}

    return model, idx_to_class

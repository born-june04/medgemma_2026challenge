from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    """
    Minimal classifier head to sit on top of a frozen image embedding.

    - If hidden_dim is None: Linear (logistic regression)
    - If hidden_dim is int: LayerNorm -> Dropout -> Linear -> GELU -> Dropout -> Linear
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
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def load_head_from_ckpt(ckpt_path: str) -> Tuple[ImageClassifier, Dict[int, str]]:
    """
    Load a saved classifier head checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

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


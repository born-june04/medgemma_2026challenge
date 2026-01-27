import hashlib
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from huggingface_hub import login

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)

from huggingface_hub import login
login(token="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")


@dataclass
class EncoderMetadata:
    name: str
    is_stub: bool
    embedding_dim: int


class ImageEncoderBase:
    """Base interface for CXR encoders.

    Clinical intent: provide a stable representation of CXR patterns for signal scoring.
    """

    def __init__(self, metadata: EncoderMetadata):
        self.metadata = metadata

    def encode(self, image_path: str) -> torch.Tensor:
        raise NotImplementedError


class StubImageEncoder(ImageEncoderBase):
    """Deterministic stub encoder used when checkpoints are unavailable."""

    def __init__(self, embedding_dim: int = 1024):
        super().__init__(EncoderMetadata(name="stub_cxr", is_stub=True, embedding_dim=embedding_dim))

    def encode(self, image_path: str) -> torch.Tensor:
        digest = _hash_file(image_path)
        rng = np.random.RandomState(int(digest[:8], 16))
        embedding = rng.normal(0.0, 1.0, size=self.metadata.embedding_dim).astype(np.float32)
        return torch.from_numpy(embedding)


class MedSigLIPImageEncoder(ImageEncoderBase):
    """Google MedSigLIP image encoder."""

    def __init__(self, checkpoint_path: str, embedding_dim: Optional[int] = None):
        super().__init__(EncoderMetadata(name="medsiglip", is_stub=False, embedding_dim=embedding_dim or 768))
        self.model_name = checkpoint_path
        self.model = None
        self.processor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._embedding_dim_initialized = False

    def _load_model(self) -> None:
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self.model.to(self._device)
            self.model.eval()

            if not self._embedding_dim_initialized:
                if hasattr(self.model, "config") and hasattr(self.model.config, "projection_dim"):
                    self.metadata.embedding_dim = int(self.model.config.projection_dim)
                self._embedding_dim_initialized = True

    def encode(self, image_path: str) -> torch.Tensor:
        self._load_model()
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            embedding = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        else:
            embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = embedding.squeeze(0)
        if self._device.type == "cuda":
            embedding = embedding.cpu()
        return embedding


def get_image_encoder(use_stub: bool, checkpoint_path: Optional[str]) -> ImageEncoderBase:
    if use_stub or not checkpoint_path:
        return StubImageEncoder()
    try:
        return MedSigLIPImageEncoder(checkpoint_path=checkpoint_path)
    except Exception:
        return StubImageEncoder()


def _hash_file(path: str, max_bytes: int = 4_000_000) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        chunk = handle.read(max_bytes)
        hasher.update(chunk)
    return hasher.hexdigest()

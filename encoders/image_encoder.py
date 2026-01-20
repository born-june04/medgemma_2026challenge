import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


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


class CXRFoundationEncoder(ImageEncoderBase):
    """Placeholder for Google CXR Foundation (HAI-DEF) encoder."""

    def __init__(self, checkpoint_path: str, embedding_dim: int = 1024):
        super().__init__(EncoderMetadata(name="cxr_hai_def", is_stub=False, embedding_dim=embedding_dim))
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError("CXR Foundation checkpoint loading is not implemented in this demo.")


def get_image_encoder(use_stub: bool, checkpoint_path: Optional[str]) -> ImageEncoderBase:
    if use_stub or not checkpoint_path:
        return StubImageEncoder()
    try:
        return CXRFoundationEncoder(checkpoint_path=checkpoint_path)
    except NotImplementedError:
        return StubImageEncoder()


def _hash_file(path: str, max_bytes: int = 4_000_000) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        chunk = handle.read(max_bytes)
        hasher.update(chunk)
    return hasher.hexdigest()

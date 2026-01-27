import hashlib
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor
from huggingface_hub import login

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)


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
                if hasattr(self.model, "config"):
                    if hasattr(self.model.config, "projection_dim"):
                        self.metadata.embedding_dim = int(self.model.config.projection_dim)
                    elif hasattr(self.model.config, "vision_config") and hasattr(
                        self.model.config.vision_config, "hidden_size"
                    ):
                        self.metadata.embedding_dim = int(self.model.config.vision_config.hidden_size)
                self._embedding_dim_initialized = True

    def encode(self, image_path: str) -> torch.Tensor:
        self._load_model()
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                embedding = self.model.get_image_features(**inputs)
            else:
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

    def gradcam(
        self,
        image_path: str,
        *,
        target: str = "embedding",
        classifier: Optional[torch.nn.Module] = None,
        class_index: Optional[int] = None,
        texts: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        target:
          - "embedding": uses image embedding norm as target score
          - "classifier": uses classifier logit for class_index (requires classifier)
          - "text": uses text-conditional logit (requires texts)
        """
        self._load_model()
        image = Image.open(image_path).convert("RGB")
        if target == "text":
            if not texts:
                raise ValueError("texts must be provided for target='text'")
            inputs = self.processor(
                text=texts,
                images=image,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self.model(
                **inputs, output_hidden_states=True, return_dict=True
            )
            hidden = outputs.vision_model_output.last_hidden_state
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)
            # Forward through vision tower with hidden states
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = vision_outputs.last_hidden_state  # [B, tokens, dim]

        tokens = hidden.shape[1]
        if tokens > 1 and int(math.sqrt(tokens - 1)) ** 2 == tokens - 1:
            patch_tokens = hidden[:, 1:, :]
            grid = int(math.sqrt(tokens - 1))
        else:
            patch_tokens = hidden
            grid = int(math.sqrt(tokens))
            if grid * grid != tokens:
                raise ValueError(f"Cannot reshape tokens ({tokens}) into square grid.")

        # Pool directly from patch tokens to keep gradient path
        pooled = patch_tokens.mean(dim=1)

        if hasattr(self.model, "visual_projection"):
            image_embeds = self.model.visual_projection(pooled)
        else:
            image_embeds = pooled

        if target == "classifier":
            if classifier is None:
                raise ValueError("classifier must be provided for target='classifier'")
            classifier = classifier.to(self._device)
            classifier.eval()
            logits = classifier(image_embeds)
            if class_index is None:
                class_index = int(torch.argmax(logits, dim=1).item())
            score = logits[0, class_index]
        elif target == "text":
            if class_index is None:
                class_index = int(torch.argmax(outputs.logits_per_image, dim=1).item())
            score = outputs.logits_per_image[0, class_index]
        else:
            score = image_embeds.norm(dim=1).sum()

        grads = torch.autograd.grad(score, patch_tokens, retain_graph=False)[0]
        weights = grads.mean(dim=2, keepdim=True)
        cam = (weights * patch_tokens).sum(dim=2)
        cam = torch.relu(cam)
        # Fallback: if CAM is all zeros, use gradient magnitude map
        if torch.max(cam).item() <= 1e-8:
            cam = grads.abs().mean(dim=2)
        cam = cam / (cam.max() + 1e-6)
        cam = cam.reshape(1, 1, grid, grid)
        cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
        return cam.squeeze().detach().cpu().numpy()

    def attention_rollout(self, image_path: str) -> np.ndarray:
        """
        Attention rollout for ViT-style models (no gradients required).
        Returns a [H, W] heatmap.
        """
        self._load_model()
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)

        outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True,
        )
        attentions = outputs.attentions  # list of [B, heads, tokens, tokens]
        if not attentions:
            raise RuntimeError("No attentions returned from vision model.")

        # Average heads and add residual connection
        attn = [a.mean(dim=1) for a in attentions]  # [B, tokens, tokens]
        tokens = attn[0].shape[-1]
        eye = torch.eye(tokens, device=attn[0].device).unsqueeze(0)
        attn = [(a + eye) / 2.0 for a in attn]

        # Rollout
        joint = attn[0]
        for a in attn[1:]:
            joint = torch.bmm(a, joint)

        # CLS to patch tokens
        if tokens > 1 and int(math.sqrt(tokens - 1)) ** 2 == tokens - 1:
            cls_to_patch = joint[:, 0, 1:]
            grid = int(math.sqrt(tokens - 1))
        else:
            cls_to_patch = joint[:, 0, :]
            grid = int(math.sqrt(tokens))
            if grid * grid != tokens:
                raise ValueError(f"Cannot reshape tokens ({tokens}) into square grid.")

        cam = cls_to_patch.reshape(1, 1, grid, grid)
        cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam = cam / (cam.max() + 1e-6)
        return cam.detach().cpu().numpy()

    def attn_gradcam(
        self,
        image_path: str,
        *,
        target: str = "text",
        classifier: Optional[torch.nn.Module] = None,
        class_index: Optional[int] = None,
        texts: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Gradient-weighted attention CAM (ViT-style).
        Uses gradients of attention weights from the last layer.
        """
        self._load_model()
        image = Image.open(image_path).convert("RGB")

        if target == "text":
            if not texts:
                raise ValueError("texts must be provided for target='text'")
            inputs = self.processor(
                text=texts,
                images=image,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )
            if class_index is None:
                class_index = int(torch.argmax(outputs.logits_per_image, dim=1).item())
            score = outputs.logits_per_image[0, class_index]
            attn = outputs.vision_model_output.attentions[-1]
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True,
            )
            attn = vision_outputs.attentions[-1]

            # Use embedding norm as default score
            hidden = vision_outputs.last_hidden_state
            pooled = hidden[:, 1:, :].mean(dim=1) if hidden.shape[1] > 1 else hidden.mean(dim=1)
            if target == "classifier":
                if classifier is None:
                    raise ValueError("classifier must be provided for target='classifier'")
                classifier = classifier.to(self._device)
                classifier.eval()
                logits = classifier(pooled)
                if class_index is None:
                    class_index = int(torch.argmax(logits, dim=1).item())
                score = logits[0, class_index]
            else:
                score = pooled.norm(dim=1).sum()

        grads = torch.autograd.grad(score, attn, retain_graph=False)[0]
        # Grad-CAM on attention: average heads and use CLS->patch attention
        weights = grads.mean(dim=1, keepdim=False)  # [B, tokens, tokens]
        attn_mean = attn.mean(dim=1, keepdim=False)
        cam = (weights * attn_mean).mean(dim=1)  # [B, tokens]

        tokens = cam.shape[-1]
        if tokens > 1 and int(math.sqrt(tokens - 1)) ** 2 == tokens - 1:
            cam = cam[:, 1:]
            grid = int(math.sqrt(tokens - 1))
        else:
            grid = int(math.sqrt(tokens))
            if grid * grid != tokens:
                raise ValueError(f"Cannot reshape tokens ({tokens}) into square grid.")

        cam = cam.reshape(1, 1, grid, grid)
        cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam = cam / (cam.max() + 1e-6)
        return cam.detach().cpu().numpy()


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

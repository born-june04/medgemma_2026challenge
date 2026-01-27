from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml

from Agent1_Audio.encoders import get_audio_encoder
from Agent1_Audio.classifiers.audio_head import load_head_from_ckpt
from Agent2_Image.encoders import get_image_encoder


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def _to_cpu_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.tensor(x)


def _run_audio_classifier(emb: torch.Tensor, ckpt_path: str) -> Optional[Dict[str, Any]]:
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        return None

    classifier, idx_to_class = load_head_from_ckpt(str(ckpt_file))
    with torch.no_grad():
        logits = classifier(emb)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        classes = [idx_to_class[i] for i in range(len(probs))]
    return {
        "pred_label": idx_to_class.get(pred_idx, str(pred_idx)),
        "pred_index": pred_idx,
        "probs": [float(p) for p in probs.tolist()],
        "classes": classes,
    }


def _fuse_image_with_audio_probs(
    image_emb: torch.Tensor, audio_probs: Optional[list[float]]
) -> torch.Tensor:
    """Late-fusion by concatenating image embedding with audio class probabilities."""
    if audio_probs is None:
        return image_emb
    probs_tensor = torch.tensor(audio_probs, dtype=image_emb.dtype)
    return torch.cat([image_emb, probs_tensor], dim=0)


def _resolve_patient_pair(patient_id: str, index_path: str) -> Tuple[str, str]:
    """Resolve (audio_path, image_path) from a CSV index by patient_id."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"pairs_index not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("patient_id") == patient_id:
                audio_path = row.get("audio_path")
                image_path = row.get("image_path") or row.get("cxr_path")
                if not audio_path or not image_path:
                    raise ValueError(
                        f"Missing audio_path/image_path for patient_id={patient_id} in {path}"
                    )
                return audio_path, image_path

    raise ValueError(f"patient_id={patient_id} not found in {path}")


def run_pipeline(
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    patient_id: Optional[str] = None,
    pairs_index: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Minimal pipeline:
      - audio embedding (+ optional classifier head)
      - image embedding
    """
    config = config or {}
    use_stub = bool(config.get("use_stub_encoders", True))

    if patient_id:
        pairs_index = pairs_index or config.get("pairs_index")
        if not pairs_index:
            raise ValueError("pairs_index must be set to use patient_id lookup.")
        audio_path, image_path = _resolve_patient_pair(patient_id, pairs_index)

    if not audio_path or not image_path:
        raise ValueError("audio_path and image_path are required (or provide patient_id).")

    audio_encoder = get_audio_encoder(use_stub, config.get("audio_checkpoint"))
    image_encoder = get_image_encoder(use_stub, config.get("image_checkpoint"))

    audio_embedding = _to_cpu_tensor(audio_encoder.encode(audio_path))
    image_embedding = _to_cpu_tensor(image_encoder.encode(image_path))

    classification_payload = None
    acfg = (config.get("audio_classifier") or {})
    if acfg.get("enabled", False):
        ckpt_path = acfg.get("checkpoint_path", "artifacts/audio_classifier_head.pt")
        try:
            classification_payload = _run_audio_classifier(audio_embedding, ckpt_path)
        except Exception as exc:
            print(f"[audio_classifier] Inference error: {exc}")

    fused_image_embedding = _fuse_image_with_audio_probs(
        image_embedding,
        classification_payload["probs"] if classification_payload else None,
    )

    return {
        "audio_embedding": audio_embedding,
        "image_embedding": image_embedding,
        "fused_image_embedding": fused_image_embedding,
        "audio_classification": classification_payload,
        "encoder_metadata": {
            "audio": audio_encoder.metadata.__dict__,
            "image": image_encoder.metadata.__dict__,
        },
    }


def run_pipeline_modes(
    audio_path: str,
    image_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {"minimal": run_pipeline(audio_path, image_path, config)}


def to_json(data: Dict[str, Any]) -> str:
    def default(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, indent=2, default=default)


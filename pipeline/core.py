from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from huggingface_hub import login

from Agent1_Audio.encoders import get_audio_encoder
from Agent1_Audio.classifiers.audio_head import load_head_from_ckpt
from Agent1_Audio.physiology.features import extract_audio_features
from Agent2_Image.encoders import get_image_encoder
from Agent2_Image.classifiers.image_head import load_head_from_ckpt as load_image_head
from Agent2_Image.utils.gradcam_utils import save_original, save_overlay_with_colorbar
from pipeline.reporting import generate_medgemma_report


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

    # Ensure HF auth is available for gated models (MedSigLIP / MedGemma).
    if config.get("hf_token") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = str(config["hf_token"])
    if os.environ.get("HF_TOKEN"):
        try:
            login(token=os.environ["HF_TOKEN"])
        except Exception:
            # Non-fatal: downstream downloads may still succeed via cached credentials.
            pass

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

    physiology_payload = None
    pcfg = (config.get("physiology") or {})
    if pcfg.get("enabled", False):
        try:
            physiology_payload = extract_audio_features(audio_path).__dict__
            out_dir = Path(str(pcfg.get("output_dir", "outputs/evidence/physiology"))) / (
                patient_id or Path(audio_path).stem
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "physiology.json").write_text(
                json.dumps(physiology_payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            physiology_payload = {"error": str(exc)}

    gradcam_payload = None
    gcfg = (config.get("gradcam") or {})
    if gcfg.get("enabled", False):
        targets = gcfg.get("targets", ["embedding"])
        output_dir = gcfg.get("output_dir", "outputs/gradcam")
        alpha = float(gcfg.get("alpha", 0.4))
        gradcam_payload = {}

        if "occlusion" in targets:
            try:
                # Build label texts from dataset_root (preferred) or fall back to audio classifier class list.
                icfg = (config.get("image_classifier") or {})
                dataset_root = icfg.get("dataset_root")
                label_texts = []
                if dataset_root:
                    root = Path(str(dataset_root))
                    if root.exists():
                        label_texts = sorted([p.name for p in root.iterdir() if p.is_dir()])
                if not label_texts and classification_payload and classification_payload.get("classes"):
                    label_texts = list(classification_payload["classes"])
                if not label_texts:
                    raise ValueError("No label texts available for occlusion. Set image_classifier.dataset_root or run audio_classifier.")

                # Predict class index via text similarity for stable patient-level label selection.
                pred = image_encoder.predict_text(image_path, texts=label_texts)
                pred_idx = int(pred["pred_index"])
                label_name = str(pred["pred_label"])

                pid = patient_id or Path(image_path).stem
                out_dir = Path(output_dir) / pid / label_name
                gradcam_payload["original"] = save_original(image_path, str(out_dir / "original.png"))
                gradcam_payload["pred"] = pred

                # Occlusion parameters (configurable)
                ocfg = (gcfg.get("occlusion") or {})
                patch_size = int(ocfg.get("patch_size", 32))
                stride = int(ocfg.get("stride", 32))
                batch_size = int(ocfg.get("batch_size", 16))

                heat_dec = image_encoder.occlusion_map(
                    image_path,
                    texts=label_texts,
                    class_index=pred_idx,
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=batch_size,
                    baseline="mean",
                    mode="decrease",
                )
                heat_inc = image_encoder.occlusion_map(
                    image_path,
                    texts=label_texts,
                    class_index=pred_idx,
                    patch_size=patch_size,
                    stride=stride,
                    batch_size=batch_size,
                    baseline="mean",
                    mode="increase",
                )

                gradcam_payload["occlusion_decrease_overlay"] = save_overlay_with_colorbar(
                    image_path,
                    heat_dec,
                    str(out_dir / "occlusion_decrease_overlay.png"),
                    alpha=alpha,
                    title="occlusion decrease = evidence-for",
                )
                gradcam_payload["occlusion_increase_overlay"] = save_overlay_with_colorbar(
                    image_path,
                    heat_inc,
                    str(out_dir / "occlusion_increase_overlay.png"),
                    alpha=alpha,
                    title="occlusion increase = distractor",
                )
            except Exception as exc:
                gradcam_payload["occlusion_error"] = str(exc)

    report_payload = None
    rcfg = (config.get("report") or {})
    if rcfg.get("enabled", False):
        try:
            pid = patient_id or Path(audio_path).stem
            report_out = Path(str(rcfg.get("output_dir", "outputs/reports"))) / pid
            model_id = str(rcfg.get("model_id") or rcfg.get("checkpoint") or "").strip()
            if not model_id:
                raise ValueError("report.model_id (or report.checkpoint) must be set")

            report_payload = generate_medgemma_report(
                {
                    "patient_id": pid,
                    "audio_path": str(audio_path),
                    "image_path": str(image_path),
                    "audio_classification": classification_payload,
                    "physiology": physiology_payload,
                    "visual_evidence": gradcam_payload,
                },
                model_id=model_id,
                out_dir=str(report_out),
                max_new_tokens=int(rcfg.get("max_new_tokens", 512)),
                temperature=float(rcfg.get("temperature", 0.2)),
            )
        except Exception as exc:
            report_payload = {"ok": False, "error": str(exc)}

    return {
        "patient_id": patient_id,
        "audio_embedding": audio_embedding,
        "image_embedding": image_embedding,
        "fused_image_embedding": fused_image_embedding,
        "audio_classification": classification_payload,
        "physiology": physiology_payload,
        "gradcam": gradcam_payload,
        "report": report_payload,
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


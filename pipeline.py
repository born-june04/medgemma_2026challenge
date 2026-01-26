import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from encoders import get_audio_encoder, get_image_encoder
from llm.placeholder import PlaceholderLLM
from physiology import compute_proxies, extract_audio_features, generate_physiology_explanations
from signals.audio_signals import audio_anomaly_score
from signals.image_signals import cxr_abnormality_score
from signals.quality import compute_audio_quality, compute_image_quality, load_audio_mono, load_image

from classifiers.audio_head import load_head_from_ckpt
import torch



DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def _is_image_path(p: str) -> bool:
    lower = p.lower()
    return lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg")


def run_pipeline(
    audio_path: str,
    image_path: str,
    intake_text: str,
    mode: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or {}
    use_stub = bool(config.get("use_stub_encoders", True))

    audio_encoder = get_audio_encoder(use_stub, config.get("audio_checkpoint"))
    image_encoder = get_image_encoder(use_stub, config.get("image_checkpoint"))

    audio_embedding = audio_encoder.encode(audio_path)
    image_embedding = image_encoder.encode(image_path)

    audio_is_image = _is_image_path(audio_path)
    if not audio_is_image:
        audio, sr = load_audio_mono(audio_path)
        audio_quality = compute_audio_quality(audio, sr)
    image = load_image(image_path)
    image_quality = compute_image_quality(image)


    # --- Optional audio classification head inference ---
    classification_payload = None
    acfg = (config.get("audio_classifier") or {})
    if acfg.get("enabled", False):
        ckpt_path = acfg.get("checkpoint_path", "artifacts/audio_classifier_head.pt")
        ckpt_file = Path(ckpt_path)
        if ckpt_file.exists():
            # Load head and run inference
            try:
                classifier, idx_to_class = load_head_from_ckpt(str(ckpt_file))
                # Ensure encoder embedding dim matches head
                D_encoder = int(audio_encoder.metadata.embedding_dim)
                D_head = classifier.net[0].normalized_shape[0] if hasattr(classifier.net[0], "normalized_shape") else None
                if D_head is not None and D_head != D_encoder:
                    # If mismatch, fail silently but informative in logs
                    print(f"[audio_classifier] Embedding dim mismatch: encoder {D_encoder} vs head {D_head}. Skipping inference.")
                else:
                    with torch.no_grad():
                        emb = audio_embedding
                        if isinstance(emb, torch.Tensor):
                            emb = emb.detach().cpu()
                        logits = classifier(emb)  # shape [1, C]
                        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [C]
                        pred_idx = int(torch.argmax(probs).item())
                        classes = [idx_to_class[i] for i in range(len(probs))]
                        classification_payload = {
                            "pred_label": idx_to_class.get(pred_idx, str(pred_idx)),
                            "pred_index": pred_idx,
                            "probs": [float(p) for p in probs.tolist()],
                            "classes": classes,
                        }
            except Exception as e:
                # Non-fatal: pipeline continues without classification
                print(f"[audio_classifier] Inference error: {e}")
        else:
            # Silent skip if no checkpoint
            pass


    signals = {
        "audio_anomaly_score": audio_anomaly_score(audio_embedding),
        "cxr_abnormality_score": cxr_abnormality_score(image_embedding),
    }

    if classification_payload is not None:
        signals["audio_classification"] = classification_payload


    features = extract_audio_features(audio_path)
    proxies = compute_proxies(features)
    explanations = generate_physiology_explanations(features, proxies)

    quality_payload = {
        "audio_duration_sec": audio_quality.duration_sec,
        "audio_clipping_fraction": audio_quality.clipping_fraction,
        "audio_noise_ratio": audio_quality.noise_ratio,
        "audio_warnings": list(audio_quality.warnings),
        "image_width": image_quality.width,
        "image_height": image_quality.height,
        "image_mode": image_quality.mode,
        "image_warnings": list(image_quality.warnings),
        "has_critical_warnings": _has_critical_warnings(audio_quality.warnings, image_quality.warnings),
    }

    physiology_payload = {
        "features": features.__dict__,
        "proxies": proxies.__dict__,
        "explanations": explanations,
    }

    structured_input = _build_structured_input(
        intake_text=intake_text,
        mode=mode,
        signals=signals,
        physiology=physiology_payload,
        quality=quality_payload,
    )

    llm = PlaceholderLLM()
    llm_output = llm.generate(structured_input)

    return {
        "mode": mode,
        "input": structured_input,
        "llm_output": llm_output,
        "encoder_metadata": {
            "audio": audio_encoder.metadata.__dict__,
            "image": image_encoder.metadata.__dict__,
        },
    }


def run_pipeline_modes(
    audio_path: str,
    image_path: str,
    intake_text: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    outputs = {}
    for mode in ("llm_only", "signals", "signals+physiology"):
        outputs[mode] = run_pipeline(audio_path, image_path, intake_text, mode, config)
    return outputs


def _build_structured_input(
    intake_text: str,
    mode: str,
    signals: Dict[str, Any],
    physiology: Dict[str, Any],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "intake_text": intake_text,
        "mode": mode,
        "quality": quality,
    }
    if mode in ("signals", "signals+physiology"):
        payload["signals"] = signals
    if mode == "signals+physiology":
        payload["physiology"] = physiology
    return payload


def _has_critical_warnings(audio_warnings, image_warnings) -> bool:
    critical_audio = "audio_too_short" in audio_warnings
    critical_image = "low_resolution" in image_warnings
    return bool(critical_audio or critical_image)


def to_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2)

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _jsonable(x: Any) -> Any:
    # Keep it robust even if torch isn't installed in the IDE environment.
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass

    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def build_report_prompt(payload: Dict[str, Any]) -> str:
    """
    Build a MedGemma instruction prompt from pipeline outputs.
    Note: this is text-only; we reference image evidence paths but do not assume the model can view images.
    """
    patient_id = payload.get("patient_id") or "unknown"
    audio_path = payload.get("audio_path") or ""
    image_path = payload.get("image_path") or ""
    audio_cls = payload.get("audio_classification") or {}
    phys = payload.get("physiology") or {}
    visual = payload.get("visual_evidence") or {}

    # Keep it compact and structured.
    return (
        "You are a clinical assistant. Write a concise medical report draft based ONLY on the provided evidence.\n"
        "Do NOT invent findings. If evidence is insufficient, say so.\n"
        "Do not repeat the instructions or the input JSON.\n"
        "Output ONLY the report (no critique, no scoring, no self-review).\n"
        "Use this structure:\n"
        "1) Impression (1-3 bullet points)\n"
        "2) Supporting evidence (bullets; cite which evidence)\n"
        "3) Caveats & recommended next steps\n\n"
        f"Patient ID: {patient_id}\n\n"
        "Original inputs (for reference; assume you cannot view images):\n"
        f"- audio/spectrogram path: {audio_path}\n"
        f"- cxr image path: {image_path}\n\n"
        "Audio classification (model output):\n"
        f"{json.dumps(_jsonable(audio_cls), ensure_ascii=False, indent=2)}\n\n"
        "Physiology features (interpretable, hypothesis-level):\n"
        f"{json.dumps(_jsonable(phys), ensure_ascii=False, indent=2)}\n\n"
        "CXR visual evidence (occlusion-based attribution files saved for human review):\n"
        f"{json.dumps(_jsonable(visual), ensure_ascii=False, indent=2)}\n"
    )


_MODEL_CACHE: Dict[str, Any] = {}


def generate_medgemma_report(
    payload: Dict[str, Any],
    *,
    model_id: str,
    out_dir: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Generate a medical report using a HF text LLM (MedGemma or compatible).
    If loading/generation fails, returns error and still saves the prompt (if out_dir is provided).
    """
    prompt = build_report_prompt(payload)

    out_paths: Dict[str, str] = {}
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "prompt.txt").write_text(prompt, encoding="utf-8")
        out_paths["prompt_txt"] = str(p / "prompt.txt")
        (p / "inputs.json").write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        out_paths["inputs_json"] = str(p / "inputs.json")

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        if model_id not in _MODEL_CACHE:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
            if token:
                tok_kwargs["token"] = token

            # Some models ship very large/complex tokenizer.json that may not parse with older tokenizers.
            # Retry with slow tokenizer as a fallback.
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **tok_kwargs)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            # Prefer device_map="auto" (requires accelerate) for 4B+ models to avoid OOM.
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if token:
                model_kwargs["token"] = token
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                import accelerate  # type: ignore  # noqa: F401

                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                # Fall back to single-device load
                pass

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            if "device_map" not in model_kwargs:
                model.to(device)
            model.eval()
            _MODEL_CACHE[model_id] = (tokenizer, model, device)

        tokenizer, model, device = _MODEL_CACHE[model_id]

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0,
                temperature=float(temperature),
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        if out_dir:
            p = Path(out_dir)
            (p / "report.txt").write_text(text, encoding="utf-8")
            out_paths["report_txt"] = str(p / "report.txt")

        return {"ok": True, "model_id": model_id, "text": text, "paths": out_paths}
    except Exception as exc:
        if out_dir:
            p = Path(out_dir)
            (p / "error.txt").write_text(str(exc), encoding="utf-8")
            out_paths["error_txt"] = str(p / "error.txt")
        return {"ok": False, "model_id": model_id, "error": str(exc), "paths": out_paths}



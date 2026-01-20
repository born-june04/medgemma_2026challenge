from typing import Any, Dict, List

from llm.adapter import LLMAdapter


class PlaceholderLLM(LLMAdapter):
    """Rule-based generator for demo use.

    Clinical intent: provide conservative explanations with explicit uncertainty.
    """

    def generate(self, structured_input: Dict[str, Any]) -> Dict[str, Any]:
        signals = structured_input.get("signals", {})
        physiology = structured_input.get("physiology", {})
        quality = structured_input.get("quality", {})

        risk_tier = _risk_tier(signals, quality)
        evidence = _evidence_summary(signals, quality)
        physio = _physiology_context(physiology)
        actions = _recommended_actions(risk_tier)
        uncertainties = _uncertainties(quality)

        return {
            "risk_tier": risk_tier,
            "evidence_summary": evidence,
            "physiological_context": physio,
            "recommended_next_actions": actions,
            "uncertainties": uncertainties,
        }


def _risk_tier(signals: Dict[str, Any], quality: Dict[str, Any]) -> str:
    if quality.get("has_critical_warnings"):
        return "insufficient_data"
    audio_score = signals.get("audio_anomaly_score", 0.0)
    cxr_score = signals.get("cxr_abnormality_score", 0.0)
    if audio_score > 0.8 or cxr_score > 0.8:
        return "high"
    if audio_score > 0.5 or cxr_score > 0.5:
        return "medium"
    return "low"


def _evidence_summary(signals: Dict[str, Any], quality: Dict[str, Any]) -> List[str]:
    summaries = []
    if "audio_anomaly_score" in signals:
        summaries.append(f"Audio anomaly score: {signals['audio_anomaly_score']:.2f} (heuristic)")
    if "cxr_abnormality_score" in signals:
        summaries.append(f"CXR abnormality score: {signals['cxr_abnormality_score']:.2f} (heuristic)")
    if quality.get("audio_warnings"):
        summaries.append(f"Audio quality flags: {', '.join(quality['audio_warnings'])}")
    if quality.get("image_warnings"):
        summaries.append(f"Image quality flags: {', '.join(quality['image_warnings'])}")
    if not summaries:
        summaries.append("No structured signal evidence provided.")
    return summaries


def _physiology_context(physiology: Dict[str, Any]) -> List[str]:
    context = physiology.get("explanations", [])
    if not context:
        return ["No physiological context provided in this mode."]
    return context


def _recommended_actions(risk_tier: str) -> List[str]:
    if risk_tier == "insufficient_data":
        return ["Repeat or obtain higher-quality recordings and imaging."]
    if risk_tier == "high":
        return [
            "Correlate with vitals and clinical exam.",
            "Consider confirmatory diagnostics per standard of care.",
        ]
    if risk_tier == "medium":
        return ["Review alongside recent symptoms and risk factors."]
    return ["Continue routine monitoring and standard clinical assessment."]


def _uncertainties(quality: Dict[str, Any]) -> List[str]:
    uncertainties = ["Heuristic scores are not calibrated for diagnosis."]
    if quality.get("has_critical_warnings"):
        uncertainties.append("Input quality limits interpretability.")
    return uncertainties

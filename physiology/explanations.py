from typing import List

from physiology.features import AudioFeatures
from physiology.proxies import PhysiologicalProxies


def generate_physiology_explanations(
    features: AudioFeatures, proxies: PhysiologicalProxies
) -> List[str]:
    """Deterministic explanations linking features to physiology.

    Clinical intent: provide transparent, rule-based context that avoids diagnosis.
    """
    explanations = []

    if features.cough_rate_per_min > 10:
        explanations.append("Frequent cough events suggest heightened airway reactivity or irritation.")
    elif features.cough_rate_per_min > 0:
        explanations.append("Occasional cough events suggest intermittent airway irritation.")
    else:
        explanations.append("No clear cough events detected; interpret cautiously.")

    if features.high_freq_energy_ratio > 0.2:
        explanations.append("High-frequency energy is elevated, consistent with turbulent airflow or secretions.")
    else:
        explanations.append("High-frequency energy is limited, suggesting less turbulent airflow.")

    if features.temporal_burstiness > 0.7:
        explanations.append("Cough timing is bursty, which can indicate episodic airway narrowing.")

    if proxies.airway_narrowing_proxy > 0.6:
        explanations.append("Spectral features lean toward narrower airway acoustics (proxy).")
    if proxies.secretion_turbulence_proxy > 0.6:
        explanations.append("Energy distribution suggests secretion-related turbulence (proxy).")
    if proxies.aerosol_generation_proxy > 0.6:
        explanations.append("Temporal pattern suggests elevated aerosol generation potential (conceptual proxy).")

    return explanations

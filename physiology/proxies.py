from dataclasses import dataclass

from physiology.features import AudioFeatures


@dataclass
class PhysiologicalProxies:
    airway_narrowing_proxy: float
    secretion_turbulence_proxy: float
    aerosol_generation_proxy: float


def compute_proxies(features: AudioFeatures) -> PhysiologicalProxies:
    """Map audio features to transparent mechanistic proxies.

    Clinical intent: make the acoustic interpretation explicit and inspectable.
    """
    airway_narrowing = _scale(features.spectral_centroid_hz, 800, 2000)
    secretion_turbulence = _scale(features.high_freq_energy_ratio, 0.05, 0.25)
    aerosol_generation = _scale(features.temporal_burstiness, 0.2, 1.0)

    return PhysiologicalProxies(
        airway_narrowing_proxy=airway_narrowing,
        secretion_turbulence_proxy=secretion_turbulence,
        aerosol_generation_proxy=aerosol_generation,
    )


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    scaled = (value - low) / (high - low)
    return float(max(0.0, min(1.0, scaled)))

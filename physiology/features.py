from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from signals.quality import load_audio_mono


@dataclass
class AudioFeatures:
    cough_rate_per_min: float
    inter_cough_interval_mean: float
    inter_cough_interval_std: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    high_freq_energy_ratio: float
    temporal_burstiness: float


def extract_audio_features(audio_path: str, target_sr: int = 16000) -> AudioFeatures:
    """Extract interpretable audio features for physiological hypotheses.

    Clinical intent: quantify cough timing and spectral structure to reason about mechanics.
    """
    audio, sr = load_audio_mono(audio_path, target_sr=target_sr)
    audio = _normalize_audio(audio)

    cough_events = _detect_cough_events(audio, sr)
    cough_rate_per_min, mean_ici, std_ici = _cough_stats(cough_events, sr)

    spectral_centroid_hz, spectral_bandwidth_hz, high_freq_energy_ratio = _spectral_features(audio, sr)
    temporal_burstiness = _burstiness(cough_events, len(audio), sr)

    return AudioFeatures(
        cough_rate_per_min=cough_rate_per_min,
        inter_cough_interval_mean=mean_ici,
        inter_cough_interval_std=std_ici,
        spectral_centroid_hz=spectral_centroid_hz,
        spectral_bandwidth_hz=spectral_bandwidth_hz,
        high_freq_energy_ratio=high_freq_energy_ratio,
        temporal_burstiness=temporal_burstiness,
    )


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio)) + 1e-8
    return audio / peak


def _detect_cough_events(audio: np.ndarray, sr: int) -> np.ndarray:
    """Simple energy-based cough detector.

    Returns indices of detected events (frame indices).
    """
    frame_size = int(0.05 * sr)
    hop = int(0.02 * sr)
    if frame_size < 1 or len(audio) < frame_size:
        return np.array([], dtype=int)

    energies = []
    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start : start + frame_size]
        energies.append(np.sqrt(np.mean(frame ** 2) + 1e-8))
    energies = np.array(energies)

    threshold = np.median(energies) + 2.5 * np.std(energies)
    peaks = np.where(energies > threshold)[0]
    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Keep only well-separated peaks
    min_gap = int(0.3 / (hop / sr))
    selected = [peaks[0]]
    for idx in peaks[1:]:
        if idx - selected[-1] >= min_gap:
            selected.append(idx)
    return np.array(selected, dtype=int)


def _cough_stats(events: np.ndarray, sr: int) -> Tuple[float, float, float]:
    if len(events) == 0:
        return 0.0, 0.0, 0.0
    event_times = events * 0.02
    if len(event_times) > 1:
        intervals = np.diff(event_times)
        return (
            float(len(events) / (event_times[-1] / 60.0 + 1e-6)),
            float(np.mean(intervals)),
            float(np.std(intervals)),
        )
    return float(len(events) / 0.5), 0.0, 0.0


def _spectral_features(audio: np.ndarray, sr: int) -> Tuple[float, float, float]:
    window = np.hanning(len(audio))
    spectrum = np.fft.rfft(audio * window)
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    mag_sum = np.sum(mag) + 1e-8
    centroid = float(np.sum(freqs * mag) / mag_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum))

    high_band = mag[freqs > 2000]
    high_freq_energy_ratio = float(np.sum(high_band) / mag_sum)
    return centroid, bandwidth, high_freq_energy_ratio


def _burstiness(events: np.ndarray, audio_len: int, sr: int) -> float:
    duration = audio_len / float(sr)
    if duration == 0:
        return 0.0
    if len(events) < 2:
        return 0.0
    event_times = events * 0.02
    intervals = np.diff(event_times)
    mean = np.mean(intervals) + 1e-8
    std = np.std(intervals)
    return float(std / mean)

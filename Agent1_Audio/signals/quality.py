from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass
class AudioQuality:
    duration_sec: float
    clipping_fraction: float
    noise_ratio: float
    warnings: Tuple[str, ...]


def load_audio_mono(path: str, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Load audio from disk, convert to mono, and optionally resample."""
    audio, sr = sf.read(path, always_2d=True)
    audio = np.mean(audio, axis=1)
    if target_sr and sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num=num_samples)
        sr = target_sr
    return audio.astype(np.float32), sr


def compute_audio_quality(audio: np.ndarray, sr: int) -> AudioQuality:
    """Lightweight audio quality checks for compatibility."""
    if audio.size == 0 or sr <= 0:
        return AudioQuality(0.0, 0.0, 0.0, ("invalid_audio",))
    duration_sec = float(audio.shape[0] / sr)
    clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))
    noise_ratio = float(np.mean(np.abs(audio) < 1e-4))
    warnings = []
    if duration_sec < 1.0:
        warnings.append("short_audio")
    if clipping_fraction > 0.01:
        warnings.append("clipping")
    return AudioQuality(duration_sec, clipping_fraction, noise_ratio, tuple(warnings))



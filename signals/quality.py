import wave
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional dependency
    sf = None


@dataclass
class AudioQuality:
    duration_sec: float
    clipping_fraction: float
    noise_ratio: float
    sample_rate: int
    warnings: Tuple[str, ...]


@dataclass
class ImageQuality:
    width: int
    height: int
    mode: str
    warnings: Tuple[str, ...]


def load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load mono audio with a minimal dependency footprint."""
    if sf is not None:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = _resample_linear(audio, sr, target_sr)
            sr = target_sr
        return audio.astype(np.float32), sr

    with wave.open(path, "rb") as handle:
        sr = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if handle.getnchannels() > 1:
            audio = audio.reshape(-1, handle.getnchannels()).mean(axis=1)
        audio /= np.iinfo(np.int16).max
        if sr != target_sr:
            audio = _resample_linear(audio, sr, target_sr)
            sr = target_sr
        return audio, sr


def compute_audio_quality(audio: np.ndarray, sr: int) -> AudioQuality:
    duration_sec = len(audio) / float(sr)
    clipping_fraction = float(np.mean(np.abs(audio) > 0.98))
    noise_ratio = _noise_ratio(audio)

    warnings = []
    if duration_sec < 1.0:
        warnings.append("audio_too_short")
    if clipping_fraction > 0.01:
        warnings.append("possible_clipping")
    if noise_ratio > 0.6:
        warnings.append("high_noise_floor")

    return AudioQuality(
        duration_sec=duration_sec,
        clipping_fraction=clipping_fraction,
        noise_ratio=noise_ratio,
        sample_rate=sr,
        warnings=tuple(warnings),
    )


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def compute_image_quality(image: Image.Image) -> ImageQuality:
    width, height = image.size
    warnings = []
    if width < 224 or height < 224:
        warnings.append("low_resolution")
    if width > 4096 or height > 4096:
        warnings.append("very_large")
    return ImageQuality(width=width, height=height, mode=image.mode, warnings=tuple(warnings))


def _noise_ratio(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    noise_floor = np.median(np.abs(audio)) + 1e-8
    return float(noise_floor / rms)


def _resample_linear(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    ratio = target_sr / float(sr)
    new_length = int(len(audio) * ratio)
    if new_length < 1:
        return audio
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, new_length)
    return np.interp(x_new, x_old, audio).astype(np.float32)

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.size == 0:
        return heat
    vmin, vmax = np.percentile(heat, [5, 95])
    if vmax - vmin < 1e-6:
        vmin, vmax = heat.min(), heat.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(heat, dtype=np.float32)
    heat = (heat - vmin) / (vmax - vmin)
    return np.clip(heat, 0.0, 1.0)


def _apply_colormap(heatmap: np.ndarray) -> Image.Image:
    """Simple colormap: blue->green->red."""
    heat = _normalize_heatmap(heatmap)
    r = np.clip(1.5 * heat - 0.5, 0, 1)
    g = np.clip(1.5 - np.abs(2 * heat - 1.0) * 2.0, 0, 1)
    b = np.clip(1.5 * (1.0 - heat) - 0.5, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def overlay_heatmap(
    image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4
) -> Image.Image:
    """
    Overlay a heatmap (H x W) on top of the input image.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")

    heat_img = _apply_colormap(heatmap).resize(image.size)

    base = image.convert("RGB")
    return Image.blend(base, heat_img, alpha=alpha)


def save_overlay(
    image_path: str,
    heatmap: np.ndarray,
    out_path: str,
    *,
    alpha: float = 0.4,
) -> str:
    """
    Save heatmap overlay to disk and return the path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)
    overlay.save(out_path)
    return out_path


def save_original(image_path: str, out_path: str) -> str:
    """
    Save original image to disk.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.open(image_path).convert("RGB").save(out_path)
    return out_path


def save_raw_heatmap(
    heatmap: np.ndarray,
    out_path: str,
) -> str:
    """
    Save raw heatmap as a grayscale PNG.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    heat = _normalize_heatmap(heatmap)
    heat = (heat * 255).astype(np.uint8)
    Image.fromarray(heat).convert("L").save(out_path)
    return out_path


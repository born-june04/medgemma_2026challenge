from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def save_colorbar(out_path: str, *, width: int = 420, height: int = 64, title: str = "importance") -> str:
    """
    Save a simple colorbar legend for the colormap used in overlays.
    - Blue: low importance
    - Red: high importance
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Gradient row
    grad = np.linspace(0.0, 1.0, num=max(2, width), dtype=np.float32)
    grad_img = _apply_colormap(grad[None, :]).resize((width, 24))

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(grad_img, (0, 22))

    draw = ImageDraw.Draw(canvas)
    # Use default PIL font (no extra dependency)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    draw.text((2, 2), f"{title} (blue=low, red=high)", fill=(0, 0, 0), font=font)
    draw.text((2, 48), "0.0", fill=(0, 0, 0), font=font)
    draw.text((width // 2 - 10, 48), "0.5", fill=(0, 0, 0), font=font)
    draw.text((width - 28, 48), "1.0", fill=(0, 0, 0), font=font)

    canvas.save(out_path)
    return out_path


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


def save_overlay_with_colorbar(
    image_path: str,
    heatmap: np.ndarray,
    out_path: str,
    *,
    alpha: float = 0.4,
    title: str = "importance",
    bar_width: int = 72,
) -> str:
    """
    Save a single figure image: (overlay | right-side colorbar).
    Color meaning: blue=low (0.0), red=high (1.0).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    base = Image.open(image_path).convert("RGB")
    ov = overlay_heatmap(base, heatmap, alpha=alpha)

    W, H = ov.size
    bar_w = max(48, int(bar_width))
    fig = Image.new("RGB", (W + bar_w, H), (255, 255, 255))
    fig.paste(ov, (0, 0))

    grad = np.linspace(1.0, 0.0, num=max(2, H), dtype=np.float32).reshape(H, 1)
    bar = _apply_colormap(grad).resize((bar_w, H))
    fig.paste(bar, (W, 0))

    draw = ImageDraw.Draw(fig)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    pad = 2
    draw.text((W + pad, pad), "1.0", fill=(0, 0, 0), font=font)
    draw.text((W + pad, H // 2 - 6), "0.5", fill=(0, 0, 0), font=font)
    draw.text((W + pad, H - 12 - pad), "0.0", fill=(0, 0, 0), font=font)
    draw.text((W + pad, 14), title, fill=(0, 0, 0), font=font)

    fig.save(out_path)
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


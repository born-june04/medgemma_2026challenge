#!/usr/bin/env python3
"""
Generate case-specific figures for Pneumothorax (agreement) and
Disagreement (Audio→Pneumonia vs CXR/CT→COVID-19) case reports.

Usage:
  python generate_case_figures.py --case pneumothorax
  python generate_case_figures.py --case disagreement
  python generate_case_figures.py                        # both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
except ImportError:
    raise ImportError("matplotlib is required: pip install matplotlib")


# ──────────────────────────────────────────────────────────────
# Style (shared with generate_figures.py)
# ──────────────────────────────────────────────────────────────
C = {
    "navy":      "#1B2838",
    "red":       "#E63946",
    "blue":      "#457B9D",
    "teal":      "#2A9D8F",
    "gold":      "#E9C46A",
    "light_bg":  "#F0F2F5",
    "bar_bg":    "#E0E4E8",
    "text":      "#1B2838",
    "muted":     "#7B8794",
    "grid":      "#D5D8DC",
    "green":     "#27AE60",
    "purple":    "#8E44AD",
    "orange":    "#E67E22",
}

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.4,
    })

_style()


# ──────────────────────────────────────────────────────────────
# Gauge bar (shared logic)
# ──────────────────────────────────────────────────────────────
def _gauge(ax, y, label, value, vmin, vmax, threshold=None,
           thresh_label=None, unit="", interpretation="", bar_color=None):
    BAR_W = 0.70
    fill = np.clip((value - vmin) / (vmax - vmin), 0, 1) * BAR_W
    if bar_color is None:
        bar_color = C["teal"]

    bar_h = 0.55
    ax.barh(y, BAR_W, height=bar_h, color=C["bar_bg"], edgecolor="none", zorder=1)
    ax.barh(y, fill, height=bar_h, color=bar_color, edgecolor="none", zorder=2, alpha=0.88)

    if threshold is not None:
        tx = np.clip((threshold - vmin) / (vmax - vmin), 0, 1) * BAR_W
        ax.plot([tx, tx], [y - 0.35, y + 0.35],
                color=C["navy"], linewidth=3, zorder=3, solid_capstyle="round")
        if thresh_label:
            ax.text(tx, y + 0.42, thresh_label, ha="center", va="bottom",
                    fontsize=12, color=C["muted"], fontweight="bold")

    ax.text(-0.02, y, label, ha="right", va="center", fontsize=15,
            fontweight="bold", color=C["text"],
            transform=ax.get_yaxis_transform())
    vx = fill + 0.02
    ax.text(vx, y, f"{value}{unit}", ha="left", va="center",
            fontsize=15, fontweight="bold", color=C["text"])

    if interpretation:
        ax.text(BAR_W + 0.08, y, interpretation, ha="left", va="center",
                fontsize=15, color=C["text"], fontweight="bold")


def _setup_gauge_panel(ax, n_gauges, title):
    ax.set_xlim(0, 1.25)
    ax.set_ylim(-1.0, n_gauges * 1.3 + 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=15, color=C["navy"])


# ──────────────────────────────────────────────────────────────
# Shared image paths (reuse real images from COVID case)
# ──────────────────────────────────────────────────────────────
SHARED_DIR = Path("docs/sample_report")
SHARED_CXR = SHARED_DIR / "cxr_original.jpeg"
SHARED_SPEC = SHARED_DIR / "spectrogram.png"


# ══════════════════════════════════════════════════════════════
# PNEUMOTHORAX CASE FIGURES
# ══════════════════════════════════════════════════════════════
def ptx_fig1_cxr(cxr_path: Path, out: Path):
    """CXR dashboard for pneumothorax case."""
    img = Image.open(cxr_path).convert("RGB")
    w, h = img.size

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    # LEFT: CXR with annotations
    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Chest X-Ray  —  Pneumothorax Evaluation", fontsize=18, pad=12)

    # Hyperlucent zone annotation (right lung)
    rect = mpatches.Rectangle((w * 0.55, h * 0.15), w * 0.35, h * 0.65,
                                fill=False, edgecolor=C["red"],
                                linewidth=3, linestyle="--")
    ax.add_patch(rect)
    ax.text(w * 0.72, h * 0.12, "Hyperlucent\nzone", ha="center",
            fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["red"],
                      alpha=0.85, edgecolor="none"))

    # Normal lung annotation
    ax.text(w * 0.30, h * 0.12, "Normal\nlung markings", ha="center",
            fontsize=12, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["green"],
                      alpha=0.75, edgecolor="none"))

    # RIGHT: Gauges
    ax_bar = fig.add_subplot(gs[1])
    _setup_gauge_panel(ax_bar, 5, "CXR Feature Gauges")

    gauges = [
        (5.6, "Opacity",          0.18, 0, 1.0, 0.40, "<0.40",  "",  "Hyper-lucent field"),
        (4.3, "Asymmetry",        0.87, 0, 1.0, 0.70, ">0.70",  "",  "Unilateral difference"),
        (3.0, "Lung Marking",     0.15, 0, 1.0, 0.30, "<0.30",  "",  "Absent peripheral marks"),
        (1.7, "Med. Shift",       0.62, 0, 1.0, 0.50, ">0.50",  "",  "Contralateral shift"),
        (0.4, "Pleural Line",     0.78, 0, 1.0, 0.60, ">0.60",  "",  "Visceral pleura visible"),
    ]
    for y, lbl, val, mn, mx, thr, tl, u, interp in gauges:
        _gauge(ax_bar, y, lbl, val, mn, mx, thr, tl, u, interp)

    ax_bar.text(0.5, -0.6, "Pneumothorax  (score: 0.93)",
                ha="center", fontsize=17, fontweight="bold", color=C["red"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF0F0",
                          edgecolor=C["red"], linewidth=2.5),
                transform=ax_bar.transData)

    fig.suptitle("Figure 1  —  CXR Agent: Pneumothorax Feature Analysis",
                 fontsize=22, fontweight="bold", y=1.02, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] PTX Fig 1 -> {out}")


def ptx_fig2_audio(spec_path: Path, out: Path):
    """Audio dashboard for pneumothorax — 'silent chest' pattern."""
    spec_img = Image.open(spec_path).convert("RGB")
    spec = np.array(spec_img)

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.imshow(spec, aspect="auto")
    ax.axis("off")
    ax.set_title("Breath Spectrogram  —  'Silent Chest' Pattern", fontsize=18, pad=12)

    # Annotations
    ax.text(128, spec.shape[0] * 0.85, "HF band: near silent",
            ha="center", fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["red"],
                      alpha=0.85, edgecolor="none"))
    ax.text(128, spec.shape[0] * 0.3, "LF: diminished sounds",
            ha="center", fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["orange"],
                      alpha=0.85, edgecolor="none"))

    # RIGHT: Gauges
    ax_bar = fig.add_subplot(gs[1])
    _setup_gauge_panel(ax_bar, 5, "Audio Feature Gauges")

    gauges = [
        (5.6, "HF Energy",     0.03, 0, 1.0, 0.05, "<0.05", "", "Near-absent (silent chest)"),
        (4.3, "Spectral Cent.", 680., 0, 4000, 2000, "<2000", "Hz", "Very low frequency"),
        (3.0, "Breath Sounds",  0.12, 0, 1.0, 0.25, "<0.25", "", "Diminished auscultation"),
        (1.7, "Cough Rate",     1.2,  0, 15,  5.0,  "<5.0",  "/m", "Minimal cough reflex"),
        (0.4, "Asymmetry",      0.81, 0, 1.0, 0.65, ">0.65", "", "Unilateral sound loss"),
    ]
    for y, lbl, val, mn, mx, thr, tl, u, interp in gauges:
        _gauge(ax_bar, y, lbl, val, mn, mx, thr, tl, u, interp, bar_color=C["blue"])

    ax_bar.text(0.5, -0.6, "Pneumothorax  (score: 0.88)",
                ha="center", fontsize=17, fontweight="bold", color=C["red"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF0F0",
                          edgecolor=C["red"], linewidth=2.5),
                transform=ax_bar.transData)

    fig.suptitle("Figure 2  —  Audio Agent: Silent Chest Analysis",
                 fontsize=22, fontweight="bold", y=1.02, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] PTX Fig 2 -> {out}")


def ptx_fig3_crossmodal(out: Path):
    """Cross-modal evidence for pneumothorax — 2-way agreement."""
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    diseases = ["Pneumothorax", "COVID-19", "Pneumonia", "Edema", "Normal", "Other"]
    audio = [0.88, 0.02, 0.04, 0.01, 0.03, 0.02]
    cxr   = [0.93, 0.01, 0.02, 0.01, 0.02, 0.01]

    x = np.arange(len(diseases))
    bw = 0.30

    ax.bar(x - bw/2, audio, bw, label="Audio (HeAR)",
           color=C["blue"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + bw/2, cxr, bw, label="CXR (MedSigLIP)",
           color=C["teal"], edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=14, fontweight="bold")
    ax.set_ylabel("Confidence Score", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("Disease Confidence by Modality", fontsize=18, fontweight="bold", pad=15)
    ax.legend(fontsize=13, framealpha=0.95, loc="upper right")
    ax.grid(axis="y", alpha=0.3, color=C["grid"])

    # RED highlight box
    highlight = mpatches.FancyBboxPatch(
        (-0.45, -0.02), 0.9, max(audio[0], cxr[0]) + 0.08,
        boxstyle="round,pad=0.03", fill=False,
        edgecolor=C["red"], linewidth=3.5, zorder=5)
    ax.add_patch(highlight)
    ax.text(0, max(audio[0], cxr[0]) + 0.10, "AGREEMENT",
            fontsize=15, fontweight="bold", color=C["red"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0",
                      edgecolor=C["red"], linewidth=1.5))

    # RIGHT: Verdict
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    card = mpatches.FancyBboxPatch(
        (0.2, 0.3), 9.6, 9.4, boxstyle="round,pad=0.5",
        facecolor=C["light_bg"], edgecolor=C["teal"], linewidth=2.5)
    ax2.add_patch(card)

    ax2.text(5, 9.2, "CROSS-MODAL VERDICT", ha="center",
             fontsize=20, fontweight="bold", color=C["navy"])

    lines = [
        (7.8, "Audio Agent:",  "Pneumothorax  (0.88)", C["blue"]),
        (6.6, "CXR Agent:",   "Pneumothorax  (0.93)", C["teal"]),
        (5.0, "CT Scan:",     "Not performed (ER)", C["muted"]),
        (3.6, "Status:",      "2-WAY AGREEMENT", C["green"]),
        (2.4, "Combined:",    "91% Confidence",  C["red"]),
    ]
    for y, lbl, val, clr in lines:
        ax2.text(1.2, y, lbl, fontsize=16, fontweight="bold", color=C["navy"])
        ax2.text(5.8, y, val, fontsize=16, fontweight="bold", color=clr)

    ax2.text(5, 1.2, "Physiological Correlation", ha="center",
             fontsize=14, fontweight="bold", color=C["navy"])
    ax2.text(5, 0.5, '"Silent chest" + unilateral hyperlucency = trapped air',
             ha="center", fontsize=13, color=C["muted"])

    fig.suptitle("Figure 3  —  Cross-Modal Evidence: Pneumothorax",
                 fontsize=22, fontweight="bold", y=1.01, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] PTX Fig 3 -> {out}")


# ══════════════════════════════════════════════════════════════
# DISAGREEMENT CASE FIGURES
# ══════════════════════════════════════════════════════════════

def dis_fig1_cxr(cxr_path: Path, out: Path):
    """CXR dashboard for disagreement case — shows COVID-like GGO."""
    img = Image.open(cxr_path).convert("RGB")
    w, h = img.size

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Chest X-Ray  —  Disagreement Case", fontsize=18, pad=12)

    rect = mpatches.Rectangle((w * 0.55, h * 0.25), w * 0.30, h * 0.45,
                                fill=False, edgecolor=C["teal"],
                                linewidth=3, linestyle="--")
    ax.add_patch(rect)
    ax.text(w * 0.70, h * 0.22, "Peripheral\nGGO", ha="center",
            fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["teal"],
                      alpha=0.85, edgecolor="none"))

    rect2 = mpatches.Rectangle((w * 0.12, h * 0.30), w * 0.30, h * 0.40,
                                 fill=False, edgecolor=C["teal"],
                                 linewidth=3, linestyle="--")
    ax.add_patch(rect2)
    ax.text(w * 0.27, h * 0.27, "Bilateral\nopacity", ha="center",
            fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["teal"],
                      alpha=0.85, edgecolor="none"))

    ax_bar = fig.add_subplot(gs[1])
    _setup_gauge_panel(ax_bar, 5, "CXR Feature Gauges")

    gauges = [
        (5.6, "Opacity",       0.61, 0, 1.0, 0.40, ">0.40", "", "Moderate GGO"),
        (4.3, "Periph. Ratio", 1.38, 0, 3.0, 1.20, ">1.20", "", "Peripheral pattern"),
        (3.0, "Symmetry",      0.84, 0, 1.0, 0.75, ">0.75", "", "Bilateral"),
        (1.7, "Consolidation", 0.42, 0, 1.0, 0.50, ">0.50", "", "Partial — ambiguous"),
        (0.4, "Texture Ent.",  0.68, 0, 1.0, 0.60, ">0.60", "", "Mixed texture"),
    ]
    for y, lbl, val, mn, mx, thr, tl, u, interp in gauges:
        _gauge(ax_bar, y, lbl, val, mn, mx, thr, tl, u, interp)

    ax_bar.text(0.5, -0.6, "CXR: COVID-19  (0.68)  |  Pneumonia  (0.22)",
                ha="center", fontsize=15, fontweight="bold", color=C["teal"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E0F7FA",
                          edgecolor=C["teal"], linewidth=2),
                transform=ax_bar.transData)

    fig.suptitle("Figure 1  —  CXR Agent Analysis (Disagreement Case)",
                 fontsize=22, fontweight="bold", y=1.02, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] DIS Fig 1 -> {out}")


def dis_fig2_audio(spec_path: Path, out: Path):
    """Audio dashboard for disagreement case — productive cough -> Pneumonia."""
    spec_img = Image.open(spec_path).convert("RGB")
    spec = np.array(spec_img)

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.imshow(spec, aspect="auto")
    ax.axis("off")
    ax.set_title("Cough Spectrogram  —  Productive Cough Pattern", fontsize=18, pad=12)

    h_s, w_s = spec.shape[0], spec.shape[1]
    ax.text(w_s * 0.5, h_s * 0.15, "Low spectral centroid (1480 Hz)",
            ha="center", fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["blue"],
                      alpha=0.85, edgecolor="none"))
    ax.text(w_s * 0.5, h_s * 0.85, "Wet cough: broadband LF energy",
            ha="center", fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["orange"],
                      alpha=0.85, edgecolor="none"))

    ax_bar = fig.add_subplot(gs[1])
    _setup_gauge_panel(ax_bar, 5, "Audio Feature Gauges")

    gauges = [
        (5.6, "Spectral Cent.", 1480., 0, 4000, 2000, "<2000", "Hz", "Wet cough range"),
        (4.3, "HF Energy",      0.19, 0, 1.0, 0.30, "<0.30", "",   "Low HF — productive"),
        (3.0, "Cough Rate",     8.4,  0, 15,  6.0,  ">6.0",  "/m", "Frequent cough"),
        (1.7, "Burstiness",     0.71, 0, 1.0, 0.55, ">0.55", "",   "Paroxysmal pattern"),
        (0.4, "LF Ratio",       0.78, 0, 1.0, 0.60, ">0.60", "",   "Sputum-laden signal"),
    ]
    for y, lbl, val, mn, mx, thr, tl, u, interp in gauges:
        _gauge(ax_bar, y, lbl, val, mn, mx, thr, tl, u, interp, bar_color=C["blue"])

    ax_bar.text(0.5, -0.6, "Audio: Pneumonia  (0.72)  |  COVID-19  (0.18)",
                ha="center", fontsize=15, fontweight="bold", color=C["blue"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD",
                          edgecolor=C["blue"], linewidth=2),
                transform=ax_bar.transData)

    fig.suptitle("Figure 2  —  Audio Agent Analysis (Disagreement Case)",
                 fontsize=22, fontweight="bold", y=1.02, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] DIS Fig 2 -> {out}")


def dis_fig3_ct(ct_path: Path, out: Path):
    """CT dashboard for disagreement case — GGO favors COVID-19."""
    img = Image.open(ct_path).convert("RGB")
    w, h = img.size

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("CT Scan (Axial)  —  COVID-19 > Pneumonia", fontsize=18, pad=12)

    ax.text(w * 0.70, h * 0.18, "Subpleural\nGGO", ha="center",
            fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["purple"],
                      alpha=0.85, edgecolor="none"))
    ax.text(w * 0.30, h * 0.80, "Bilateral\ndistribution", ha="center",
            fontsize=13, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["purple"],
                      alpha=0.85, edgecolor="none"))

    ax_bar = fig.add_subplot(gs[1])
    _setup_gauge_panel(ax_bar, 5, "CT Feature Gauges")

    gauges = [
        (5.6, "GGO Extent",    0.64, 0, 1.0, 0.40, ">0.40", "", "Widespread GGO"),
        (4.3, "Crazy-Paving",  0.41, 0, 1.0, 0.30, ">0.30", "", "Present — viral sign"),
        (3.0, "Periph. Dist.", 0.78, 0, 1.0, 0.60, ">0.60", "", "Subpleural dominant"),
        (1.7, "Bilaterality",  0.89, 0, 1.0, 0.70, ">0.70", "", "Both lungs"),
        (0.4, "Consolidation", 0.35, 0, 1.0, 0.50, "<0.50", "", "Minimal — not bacterial"),
    ]
    for y, lbl, val, mn, mx, thr, tl, u, interp in gauges:
        _gauge(ax_bar, y, lbl, val, mn, mx, thr, tl, u, interp, bar_color=C["purple"])

    ax_bar.text(0.5, -0.6, "CT: COVID-19  (0.81)  |  Pneumonia  (0.12)",
                ha="center", fontsize=15, fontweight="bold", color=C["purple"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3E5F5",
                          edgecolor=C["purple"], linewidth=2),
                transform=ax_bar.transData)

    fig.suptitle("Figure 3  —  CT Agent Analysis (Disagreement Case)",
                 fontsize=22, fontweight="bold", y=1.02, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] DIS Fig 3 -> {out}")


def dis_fig4_crossmodal(out: Path):
    """Cross-modal bar chart showing disagreement between agents."""
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    diseases = ["COVID-19", "Pneumonia", "TB", "Pneumothorax", "Edema", "Other"]
    audio = [0.18, 0.72, 0.04, 0.01, 0.03, 0.02]
    cxr   = [0.68, 0.22, 0.04, 0.01, 0.03, 0.02]
    ct    = [0.81, 0.12, 0.03, 0.00, 0.02, 0.02]

    x = np.arange(len(diseases))
    bw = 0.22

    ax.bar(x - bw, audio, bw, label="Audio (HeAR)",
           color=C["blue"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x,      cxr,   bw, label="CXR (MedSigLIP)",
           color=C["teal"], edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + bw, ct,    bw, label="CT (MedSigLIP)",
           color=C["purple"], edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=14, fontweight="bold")
    ax.set_ylabel("Confidence Score", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-Disease Confidence by Modality", fontsize=18,
                 fontweight="bold", pad=15)
    ax.legend(fontsize=13, framealpha=0.95, loc="upper right")
    ax.grid(axis="y", alpha=0.3, color=C["grid"])

    highlight1 = mpatches.FancyBboxPatch(
        (-0.45, -0.02), 0.9, 0.90,
        boxstyle="round,pad=0.03", fill=False,
        edgecolor=C["orange"], linewidth=3.5, linestyle="--", zorder=5)
    ax.add_patch(highlight1)

    highlight2 = mpatches.FancyBboxPatch(
        (0.55, -0.02), 0.9, 0.82,
        boxstyle="round,pad=0.03", fill=False,
        edgecolor=C["orange"], linewidth=3.5, linestyle="--", zorder=5)
    ax.add_patch(highlight2)

    ax.text(0.5, 0.95, "SPLIT",
            fontsize=14, fontweight="bold", color=C["orange"], ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1",
                      edgecolor=C["orange"], linewidth=1.5))

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    card = mpatches.FancyBboxPatch(
        (0.2, 0.3), 9.6, 9.4, boxstyle="round,pad=0.5",
        facecolor="#FFF8E1", edgecolor=C["orange"], linewidth=3)
    ax2.add_patch(card)

    ax2.text(5, 9.2, "DISAGREEMENT DETECTED", ha="center",
             fontsize=19, fontweight="bold", color=C["red"])

    lines = [
        (7.8, "Audio Agent:", "Pneumonia  (0.72)", C["blue"]),
        (6.6, "CXR Agent:",  "COVID-19  (0.68)", C["teal"]),
        (5.4, "CT Agent:",   "COVID-19  (0.81)", C["purple"]),
        (4.0, "Status:",     "2:1 SPLIT", C["orange"]),
    ]
    for y, lbl, val, clr in lines:
        ax2.text(1.2, y, lbl, fontsize=16, fontweight="bold", color=C["navy"])
        ax2.text(5.8, y, val, fontsize=16, fontweight="bold", color=clr)

    ax2.text(5, 2.6, "Resolution", ha="center",
             fontsize=16, fontweight="bold", color=C["navy"])
    ax2.text(5, 1.8, "Weighted vote: COVID-19 (73%)",
             ha="center", fontsize=14, color=C["text"], fontweight="bold")
    ax2.text(5, 1.0, "FLAG: NEEDS CONFIRMATION",
             ha="center", fontsize=14, fontweight="bold", color=C["red"])

    fig.suptitle("Figure 4  —  Cross-Modal Disagreement Summary",
                 fontsize=22, fontweight="bold", y=1.01, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] DIS Fig 4 -> {out}")


def dis_fig5_resolution(out: Path):
    """Academic-style disagreement resolution with LaTeX-rendered equations."""
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.2, 1.8, 1.0], hspace=0.25)

    # ─── Panel A: Problem Definition ───
    ax_a = fig.add_subplot(gs[0])
    ax_a.axis("off")
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)

    ax_a.text(0.3, 9.0, "A.  Problem Definition", fontsize=18,
              fontweight="bold", color=C["navy"])
    ax_a.axhline(y=8.3, xmin=0.02, xmax=0.98, color=C["grid"], linewidth=1)

    defn = [
        r"Let $\mathcal{M} = \{m_1, m_2, m_3\}$ denote modalities (Audio, CXR, CT)  and  $\mathcal{D} = \{d_1, \ldots, d_9\}$  the disease classes.",
        r"Each agent $m$ produces $\mathbf{p}_m = \mathrm{softmax}\!\left(\mathbf{W}_m \cdot \phi_m(\mathbf{x}_m) + \mathbf{b}_m\right) \in \Delta^{|\mathcal{D}|-1}$",
        r"where $\phi_m$ is a frozen encoder (HeAR or MedSigLIP) and $(\mathbf{W}_m, \mathbf{b}_m)$ are learned MLP parameters.",
    ]
    for i, line in enumerate(defn):
        ax_a.text(0.5, 6.5 - i * 1.8, line, fontsize=13.5,
                  color=C["text"], math_fontfamily="cm")

    # ─── Panel B: Weighted Score Aggregation ───
    ax_b = fig.add_subplot(gs[1])
    ax_b.axis("off")
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)

    ax_b.text(0.3, 9.5, "B.  Weighted Score Aggregation", fontsize=18,
              fontweight="bold", color=C["navy"])
    ax_b.axhline(y=8.8, xmin=0.02, xmax=0.98, color=C["grid"], linewidth=1)

    ax_b.text(5.0, 7.2,
              r"$S(d) = \sum_{m \in \mathcal{M}} w_m \cdot p_{m}(d)$",
              fontsize=24, ha="center", color=C["text"], math_fontfamily="cm")

    ax_b.text(0.5, 5.2, "Modality specificity weights:", fontsize=14, color=C["muted"])

    # Table-style weights
    weights = [
        ("CT",    r"$w_{\mathrm{ct}} = 1.2$",    "Highest spatial resolution"),
        ("CXR",   r"$w_{\mathrm{cxr}} = 1.0$",   "Standard radiographic baseline"),
        ("Audio", r"$w_{\mathrm{audio}} = 0.8$",  "Complementary, lower specificity"),
    ]
    for i, (m, w, r) in enumerate(weights):
        y_pos = 4.0 - i * 1.2
        ax_b.text(1.5, y_pos, m, fontsize=14, fontweight="bold", color=C["navy"])
        ax_b.text(3.5, y_pos, w, fontsize=14, color=C["text"], math_fontfamily="cm")
        ax_b.text(6.5, y_pos, r, fontsize=13, color=C["muted"])

    # ─── Panel C: Decision Rule + Worked Example ───
    ax_c = fig.add_subplot(gs[2])
    ax_c.axis("off")
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)

    ax_c.text(0.3, 9.7, "C.  Decision Rule and Worked Example", fontsize=18,
              fontweight="bold", color=C["navy"])
    ax_c.axhline(y=9.0, xmin=0.02, xmax=0.98, color=C["grid"], linewidth=1)

    ax_c.text(5.0, 8.0,
              r"$\hat{d} = \arg\max_{d \in \mathcal{D}} \; S(d)$",
              fontsize=24, ha="center", color=C["text"], math_fontfamily="cm")

    # Worked example in a box
    box = mpatches.FancyBboxPatch(
        (0.3, 2.2), 9.4, 5.0, boxstyle="round,pad=0.3",
        facecolor="#F8F9FA", edgecolor=C["grid"], linewidth=1.5)
    ax_c.add_patch(box)

    example_lines = [
        (6.6, r"$S(\mathrm{COVID\text{-}19})$", "="),
        (5.8, r"$w_{\mathrm{audio}} \cdot 0.18 \;+\; w_{\mathrm{cxr}} \cdot 0.68 \;+\; w_{\mathrm{ct}} \cdot 0.81$", ""),
        (5.0, r"$= 0.8 \times 0.18 + 1.0 \times 0.68 + 1.2 \times 0.81 = \mathbf{1.796}$", ""),
        (3.8, r"$S(\mathrm{Pneumonia})$", "="),
        (3.0, r"$w_{\mathrm{audio}} \cdot 0.72 \;+\; w_{\mathrm{cxr}} \cdot 0.22 \;+\; w_{\mathrm{ct}} \cdot 0.12$", ""),
        (2.2, r"$= 0.8 \times 0.72 + 1.0 \times 0.22 + 1.2 \times 0.12 = \mathbf{0.940}$", ""),
    ]
    for y_pos, txt, _ in example_lines:
        x_pos = 1.0 if txt.startswith(r"$S(") else 2.2
        ax_c.text(x_pos, y_pos + 0.3, txt, fontsize=14,
                  color=C["text"], math_fontfamily="cm")

    # Result
    ax_c.text(5.0, 1.2,
              r"$\hat{d} = \mathrm{COVID\text{-}19}$,    "
              r"Normalized: $\frac{S(\mathrm{COVID\text{-}19})}{S(\mathrm{COVID\text{-}19}) + S(\mathrm{Pneumonia})} = \frac{1.796}{2.736} = 65.6\%$",
              fontsize=14, ha="center", color=C["green"], fontweight="bold",
              math_fontfamily="cm")

    # ─── Panel D: Flagging Criterion ───
    ax_d = fig.add_subplot(gs[3])
    ax_d.axis("off")
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)

    ax_d.text(0.3, 9.0, "D.  Confidence Flagging Criterion", fontsize=18,
              fontweight="bold", color=C["navy"])
    ax_d.axhline(y=8.3, xmin=0.02, xmax=0.98, color=C["grid"], linewidth=1)

    ax_d.text(5.0, 6.2,
              r"$\mathrm{FLAG} = \mathbb{1}\!\left["
              r"\exists\, m_i, m_j :\; "
              r"\arg\max_d p_{m_i}\!(d) \neq \arg\max_d p_{m_j}\!(d)"
              r"\;\wedge\;"
              r"\max_d p_{m_i}\!(d) > \tau"
              r"\;\wedge\;"
              r"\max_d p_{m_j}\!(d) > \tau"
              r"\right]$",
              fontsize=13, ha="center", color=C["text"], math_fontfamily="cm")

    ax_d.text(5.0, 4.2,
              r"Threshold: $\tau = 0.5$ (minimum confidence to trigger flag)",
              fontsize=14, ha="center", color=C["muted"], math_fontfamily="cm")

    ax_d.text(0.5, 2.5, "This case:", fontsize=14, fontweight="bold", color=C["navy"])
    ax_d.text(0.5, 1.2,
              r"Audio: $\arg\max = \mathrm{Pneumonia}$ (0.72 > 0.5)"
              r"  $\neq$  CXR: $\arg\max = \mathrm{COVID\text{-}19}$ (0.68 > 0.5)"
              r"  $\;\Rightarrow\;$  FLAG = 1",
              fontsize=13, color=C["red"], fontweight="bold", math_fontfamily="cm")

    fig.suptitle("Figure 5  —  Disagreement Resolution: Formal Specification",
                 fontsize=22, fontweight="bold", y=0.98, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] DIS Fig 5 -> {out}")


# ══════════════════════════════════════════════════════════════
# PTX FIGURE 4: MedGemma Integrated Report
# ══════════════════════════════════════════════════════════════
def ptx_fig4_report(cxr_path: Path, spec_path: Path, out: Path):
    """MedGemma integrated report for the pneumothorax agreement case."""

    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(6, 3, height_ratios=[0.6, 1.8, 2.4, 2.0, 1.8, 0.4],
                           hspace=0.20, wspace=0.25)

    # ─── Row 0: Title header ───
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis("off")
    ax_hdr.set_xlim(0, 10); ax_hdr.set_ylim(0, 2)
    hdr_box = mpatches.FancyBboxPatch(
        (0, 0), 10, 2, boxstyle="round,pad=0.3",
        facecolor=C["navy"], edgecolor="none")
    ax_hdr.add_patch(hdr_box)
    ax_hdr.text(5, 1.3, "MULTIMODAL PULMONARY DIAGNOSTIC REPORT",
                ha="center", fontsize=24, fontweight="bold", color="white")
    ax_hdr.text(5, 0.5, "Patient: PTX_DEMO_001  |  MedGemma 1.5-4B-IT  |  2 Modalities (CXR + Audio)",
                ha="center", fontsize=14, color="#AAB8C2")

    # ─── Row 1: Input images ───
    images = [
        (cxr_path,  "Chest X-Ray (PA)", gs[1, 0]),
        (spec_path, "Cough Spectrogram", gs[1, 1]),
    ]
    for path, title, gs_pos in images:
        ax = fig.add_subplot(gs_pos)
        if path.exists():
            im = Image.open(path).convert("RGB")
            ax.imshow(im)
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

    ax_empty = fig.add_subplot(gs[1, 2])
    ax_empty.axis("off")
    ax_empty.text(0.5, 0.5, "CT not ordered\n(clinical decision)",
                  ha="center", va="center", fontsize=14, color=C["muted"],
                  transform=ax_empty.transAxes)

    # ─── Row 2: Per-agent findings ───
    agent_data = [
        ("Audio Evidence (HeAR)", C["blue"], [
            "Pneumothorax (76% conf.)",
            "L1: Structural cluster (88%)",
            "L2: Diminished breath sounds",
            "L3: Pneumothorax (0.76)",
            "Ruled out: COVID-19",
        ]),
        ("CXR Evidence (MedSigLIP)", C["teal"], [
            "Pneumothorax (92% conf.)",
            "L1: Air/collapse pattern (95%)",
            "L2: Visible pleural line, absent lung markings",
            "L3: Pneumothorax (0.92)",
            "Ruled out: Pleural Effusion",
        ]),
    ]
    for col, (title, color, bullets) in enumerate(agent_data):
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        card = mpatches.FancyBboxPatch(
            (0.2, 0.2), 9.6, 9.4, boxstyle="round,pad=0.4",
            facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(card)
        ax.text(5, 8.8, title, ha="center", fontsize=15, fontweight="bold", color=color)
        ax.axhline(y=8.2, xmin=0.1, xmax=0.9, color=C["grid"], linewidth=1)
        for i, bullet in enumerate(bullets):
            y = 7.4 - i * 1.4
            is_ruled = "Ruled out" in bullet
            ax.text(1.0, y, "-" if not is_ruled else "x", fontsize=14,
                    fontweight="bold", color=C["muted"] if is_ruled else color)
            ax.text(1.6, y, bullet, fontsize=13,
                    color=C["muted"] if is_ruled else C["text"])

    ax_agree = fig.add_subplot(gs[2, 2])
    ax_agree.axis("off")
    ax_agree.set_xlim(0, 10); ax_agree.set_ylim(0, 10)
    agree_card = mpatches.FancyBboxPatch(
        (0.2, 0.2), 9.6, 9.4, boxstyle="round,pad=0.4",
        facecolor="#E8F5E9", edgecolor=C["green"], linewidth=2)
    ax_agree.add_patch(agree_card)
    ax_agree.text(5, 8.0, "✓ 2-WAY\nAGREEMENT", ha="center", fontsize=20,
                  fontweight="bold", color=C["green"])
    ax_agree.text(5, 5.5, "Both agents converge\non Pneumothorax\nNo flag raised",
                  ha="center", fontsize=14, color=C["text"])

    # ─── Row 3: Impression ───
    ax_imp = fig.add_subplot(gs[3, :])
    ax_imp.axis("off")
    ax_imp.set_xlim(0, 10); ax_imp.set_ylim(0, 10)

    imp_box = mpatches.FancyBboxPatch(
        (0.1, 5.0), 9.8, 4.8, boxstyle="round,pad=0.5",
        facecolor="#FFF8F0", edgecolor=C["red"], linewidth=2.5)
    ax_imp.add_patch(imp_box)
    ax_imp.text(5, 9.2, "IMPRESSION", ha="center",
                fontsize=20, fontweight="bold", color=C["red"])
    for i, line in enumerate([
        "High probability of pneumothorax",
        "CXR: visible pleural line with absent peripheral lung markings",
        "Audio: diminished breath sounds confirming air trapping",
    ]):
        ax_imp.text(0.8, 8.0 - i * 1.0, line, fontsize=15,
                    color=C["text"], fontweight="bold" if i == 0 else "normal")

    agr_box = mpatches.FancyBboxPatch(
        (0.1, 0.3), 9.8, 3.8, boxstyle="round,pad=0.5",
        facecolor="#E8F5E9", edgecolor=C["green"], linewidth=2.5)
    ax_imp.add_patch(agr_box)
    ax_imp.text(5, 3.5, "2-WAY MULTIMODAL AGREEMENT", ha="center",
                fontsize=18, fontweight="bold", color=C["green"])
    for i, line in enumerate([
        "Audio: diminished breath sounds, reduced spectral energy → air space",
        "CXR: unilateral hyperlucency, visible pleural line, absent markings",
    ]):
        ax_imp.text(0.8, 2.5 - i * 0.85, line, fontsize=14, color=C["text"])

    # ─── Row 4: Next steps ───
    ax_cav = fig.add_subplot(gs[4, :])
    ax_cav.axis("off")
    ax_cav.set_xlim(0, 10); ax_cav.set_ylim(0, 10)

    ax_cav.text(0.3, 9.0, "RECOMMENDED NEXT STEPS", fontsize=17,
                fontweight="bold", color=C["gold"])
    ax_cav.axhline(y=8.4, xmin=0.02, xmax=0.98, color=C["grid"], linewidth=1)
    steps = [
        "!  Assess pneumothorax size — if >2cm or symptomatic, chest tube indicated",
        "!  Urgent surgical consultation if tension pneumothorax suspected",
        ">  CT chest for precise sizing if clinical picture ambiguous",
        ">  Serial CXR in 6-12 hours if conservative management chosen",
    ]
    for i, step in enumerate(steps):
        sym = step[0]
        text = step[3:]
        color = C["red"] if sym == "!" else C["blue"]
        ax_cav.text(0.6, 7.2 - i * 1.2, sym, fontsize=14, fontweight="bold", color=color)
        ax_cav.text(1.1, 7.2 - i * 1.2, text, fontsize=13, color=C["text"])

    # ─── Row 5: Footer ───
    ax_ft = fig.add_subplot(gs[5, :])
    ax_ft.axis("off")
    ax_ft.set_xlim(0, 10); ax_ft.set_ylim(0, 1)
    ft_box = mpatches.FancyBboxPatch(
        (0, 0), 10, 1, boxstyle="round,pad=0.2",
        facecolor=C["light_bg"], edgecolor=C["grid"], linewidth=1)
    ax_ft.add_patch(ft_box)
    ax_ft.text(5, 0.5,
               "FOR RESEARCH PURPOSES ONLY  |  Google HAI-DEF:  HeAR  +  MedSigLIP  +  MedGemma",
               ha="center", fontsize=12, color=C["muted"])

    fig.suptitle("Figure 4  —  Integrated Clinical Report (MedGemma Output)",
                 fontsize=22, fontweight="bold", y=0.995, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] PTX Fig 4 -> {out}")


# ══════════════════════════════════════════════════════════════
# DIS FIGURE 6: MedGemma Integrated Report
# ══════════════════════════════════════════════════════════════
def dis_fig6_report(cxr_path: Path, spec_path: Path, ct_path: Path, out: Path):
    """MedGemma integrated report for the disagreement case."""

    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(6, 3, height_ratios=[0.6, 1.8, 2.4, 2.0, 1.8, 0.4],
                           hspace=0.20, wspace=0.25)

    # ─── Row 0: Title header ───
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis("off")
    ax_hdr.set_xlim(0, 10); ax_hdr.set_ylim(0, 2)
    hdr_box = mpatches.FancyBboxPatch(
        (0, 0), 10, 2, boxstyle="round,pad=0.3",
        facecolor=C["navy"], edgecolor="none")
    ax_hdr.add_patch(hdr_box)
    ax_hdr.text(5, 1.3, "MULTIMODAL PULMONARY DIAGNOSTIC REPORT",
                ha="center", fontsize=24, fontweight="bold", color="white")
    ax_hdr.text(5, 0.5, "Patient: DISAGREE_DEMO_001  |  MedGemma 1.5-4B-IT  |  3 Modalities",
                ha="center", fontsize=14, color="#AAB8C2")

    # ─── Row 1: Input images ───
    images = [
        (cxr_path,  "Chest X-Ray (PA)", gs[1, 0]),
        (spec_path, "Cough Spectrogram", gs[1, 1]),
        (ct_path,   "CT Scan (Axial)",   gs[1, 2]),
    ]
    for path, title, gs_pos in images:
        ax = fig.add_subplot(gs_pos)
        if path.exists():
            im = Image.open(path).convert("RGB")
            ax.imshow(im)
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

    # ─── Row 2: Per-agent findings ───
    agent_data = [
        ("Audio Evidence (HeAR)", C["blue"], [
            "Pneumonia (72% conf.)",
            "L1: Infectious cluster (85%)",
            "L2: Wet cough — 1480 Hz",
            "L3: Pneumonia (0.72)",
            "Ruled out: COVID-19 (0.18)",
        ]),
        ("CXR Evidence (MedSigLIP)", C["teal"], [
            "COVID-19 (68% conf.)",
            "L1: Opacity cluster (91%)",
            "L2: Peripheral GGO pattern",
            "L3: COVID-19 (0.68)",
            "Ruled out: Pneumonia (0.22)",
        ]),
        ("CT Evidence (MedSigLIP)", C["purple"], [
            "COVID-19 (81% conf.)",
            "L1: GGO pattern (93%)",
            "L2: Bilateral subpleural GGO",
            "L3: COVID-19 (0.81)",
            "Ruled out: Pneumonia (0.12)",
        ]),
    ]
    for col, (title, color, bullets) in enumerate(agent_data):
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        card = mpatches.FancyBboxPatch(
            (0.2, 0.2), 9.6, 9.4, boxstyle="round,pad=0.4",
            facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(card)
        ax.text(5, 8.8, title, ha="center", fontsize=15, fontweight="bold", color=color)
        ax.axhline(y=8.2, xmin=0.1, xmax=0.9, color=C["grid"], linewidth=1)
        for i, bullet in enumerate(bullets):
            y = 7.4 - i * 1.4
            is_ruled = "Ruled out" in bullet
            ax.text(1.0, y, "-" if not is_ruled else "x", fontsize=14,
                    fontweight="bold", color=C["muted"] if is_ruled else color)
            ax.text(1.6, y, bullet, fontsize=13,
                    color=C["muted"] if is_ruled else C["text"])

    # ─── Row 3: Impression + Disagreement Alert ───
    ax_imp = fig.add_subplot(gs[3, :])
    ax_imp.axis("off")
    ax_imp.set_xlim(0, 10); ax_imp.set_ylim(0, 10)

    # Warning banner
    warn_box = mpatches.FancyBboxPatch(
        (0.1, 5.0), 9.8, 4.8, boxstyle="round,pad=0.5",
        facecolor="#FFF3E0", edgecolor=C["orange"], linewidth=2.5)
    ax_imp.add_patch(warn_box)
    ax_imp.text(5, 9.2, "⚠️  IMPRESSION — DISAGREEMENT DETECTED", ha="center",
                fontsize=20, fontweight="bold", color=C["orange"])
    for i, line in enumerate([
        "Primary hypothesis: COVID-19 pneumonitis (65.6% weighted confidence)",
        "Audio agent dissents: productive cough → Pneumonia (0.72)",
        "Possible co-infection: COVID-19 with secondary bacterial superinfection",
    ]):
        ax_imp.text(0.8, 8.0 - i * 1.0, line, fontsize=15,
                    color=C["text"], fontweight="bold" if i == 0 else "normal")

    # Resolution card
    res_box = mpatches.FancyBboxPatch(
        (0.1, 0.3), 9.8, 3.8, boxstyle="round,pad=0.5",
        facecolor="#FFF8E1", edgecolor=C["orange"], linewidth=2.5)
    ax_imp.add_patch(res_box)
    ax_imp.text(5, 3.5, "WEIGHTED VOTE RESOLUTION", ha="center",
                fontsize=18, fontweight="bold", color=C["orange"])
    for i, line in enumerate([
        "S(COVID-19) = 0.8×0.18 + 1.0×0.68 + 1.2×0.81 = 1.796",
        "S(Pneumonia) = 0.8×0.72 + 1.0×0.22 + 1.2×0.12 = 0.940",
        "Decision: COVID-19 (65.6%)  |  FLAG: NEEDS CONFIRMATION",
    ]):
        color = C["red"] if "FLAG" in line else C["text"]
        weight = "bold" if "FLAG" in line else "normal"
        ax_imp.text(0.8, 2.5 - i * 0.85, line, fontsize=14, color=color, fontweight=weight)

    # ─── Row 4: Next steps ───
    ax_cav = fig.add_subplot(gs[4, :])
    ax_cav.axis("off")
    ax_cav.set_xlim(0, 10); ax_cav.set_ylim(0, 10)

    ax_cav.text(0.3, 9.0, "RECOMMENDED NEXT STEPS", fontsize=17,
                fontweight="bold", color=C["gold"])
    ax_cav.axhline(y=8.4, xmin=0.02, xmax=0.98, color=C["grid"], linewidth=1)
    steps = [
        "!  RT-PCR for SARS-CoV-2 — resolves the diagnostic split",
        "!  Sputum culture + Gram stain — identifies bacterial co-infection",
        ">  Procalcitonin level — biomarker for bacterial vs viral",
        ">  Blood cultures × 2 sets — if superinfection suspected",
        ">  Repeat CT in 5-7 days — track progression pattern",
    ]
    for i, step in enumerate(steps):
        sym = step[0]
        text = step[3:]
        color = C["red"] if sym == "!" else C["blue"]
        ax_cav.text(0.6, 7.2 - i * 1.0, sym, fontsize=14, fontweight="bold", color=color)
        ax_cav.text(1.1, 7.2 - i * 1.0, text, fontsize=13, color=C["text"])

    # ─── Row 5: Footer ───
    ax_ft = fig.add_subplot(gs[5, :])
    ax_ft.axis("off")
    ax_ft.set_xlim(0, 10); ax_ft.set_ylim(0, 1)
    ft_box = mpatches.FancyBboxPatch(
        (0, 0), 10, 1, boxstyle="round,pad=0.2",
        facecolor=C["light_bg"], edgecolor=C["grid"], linewidth=1)
    ax_ft.add_patch(ft_box)
    ax_ft.text(5, 0.5,
               "FOR RESEARCH PURPOSES ONLY  |  Google HAI-DEF:  HeAR  +  MedSigLIP  +  MedGemma",
               ha="center", fontsize=12, color=C["muted"])

    fig.suptitle("Figure 6  —  Integrated Clinical Report (MedGemma Output)",
                 fontsize=22, fontweight="bold", y=0.995, color=C["navy"])
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  [OK] DIS Fig 6 -> {out}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["pneumothorax", "disagreement", "all"],
                        default="all")
    parser.add_argument("--output-dir", default="docs/sample_report")
    args = parser.parse_args()
    out = Path(args.output_dir)

    if args.case in ("pneumothorax", "all"):
        ptx_dir = out / "pneumothorax"
        ptx_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*55}")
        print(f"  Generating Pneumothorax Case Figures")
        print(f"{'='*55}\n")

        ptx_fig1_cxr(SHARED_CXR, ptx_dir / "fig1_cxr_dashboard.png")
        ptx_fig2_audio(SHARED_SPEC, ptx_dir / "fig2_audio_dashboard.png")
        ptx_fig3_crossmodal(ptx_dir / "fig3_cross_modal.png")
        ptx_fig4_report(SHARED_CXR, SHARED_SPEC,
                        ptx_dir / "fig4_medgemma_report.png")

    if args.case in ("disagreement", "all"):
        dis_dir = out / "disagreement"
        dis_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*55}")
        print(f"  Generating Disagreement Case Figures")
        print(f"{'='*55}\n")

        ct_path = SHARED_DIR / "ct_original.png"
        dis_fig1_cxr(SHARED_CXR, dis_dir / "fig1_cxr_dashboard.png")
        dis_fig2_audio(SHARED_SPEC, dis_dir / "fig2_audio_dashboard.png")
        dis_fig3_ct(ct_path, dis_dir / "fig3_ct_dashboard.png")
        dis_fig4_crossmodal(dis_dir / "fig4_disagreement_bars.png")
        dis_fig5_resolution(dis_dir / "fig5_resolution_formal.png")
        dis_fig6_report(SHARED_CXR, SHARED_SPEC, ct_path,
                        dis_dir / "fig6_medgemma_report.png")

    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate ideal sample pipeline outputs for demonstration purposes.

This script uses the hierarchical analysis engines directly with
synthetic feature vectors to produce example outputs that showcase
the full pipeline reasoning process — no GPU or trained models required.

Usage:
    python generate_sample_outputs.py [--output-dir docs/sample_outputs]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Agent1_Audio.physiology.features import AudioFeatures
from Agent1_Audio.physiology.analyzer import HierarchicalPhysiologyAnalyzer
from Agent2_Image.physiology.features import CXRFeatures
from Agent2_Image.physiology.analyzer import HierarchicalCXRAnalyzer


# ---------------------------------------------------------------------------
# Synthetic feature profiles — designed to produce ideal outputs
# ---------------------------------------------------------------------------

PROFILES = {
    "covid19": {
        "label": "1. COVID-19",
        "audio": AudioFeatures(
            spectral_centroid=2340.0,
            spectral_bandwidth=1850.0,
            hf_energy_ratio=0.38,
            temporal_burstiness=0.52,
            cough_rate=8.2,
            zero_crossing_rate=0.12,
            mean_intensity_db=-22.5,
        ),
        "cxr": CXRFeatures(
            mean_opacity=0.58,
            peripheral_central_ratio=1.52,
            texture_entropy=0.72,
            texture_homogeneity=0.31,
            symmetry_ratio=0.91,
            upper_zone_ratio=0.22,
            mid_zone_ratio=0.41,
            lower_zone_ratio=0.37,
        ),
    },
    "tuberculosis": {
        "label": "5. Tuberculosis",
        "audio": AudioFeatures(
            spectral_centroid=1680.0,
            spectral_bandwidth=1400.0,
            hf_energy_ratio=0.18,
            temporal_burstiness=0.74,
            cough_rate=6.1,
            zero_crossing_rate=0.09,
            mean_intensity_db=-28.0,
        ),
        "cxr": CXRFeatures(
            mean_opacity=0.52,
            peripheral_central_ratio=0.95,
            texture_entropy=0.55,
            texture_homogeneity=0.38,
            symmetry_ratio=0.62,
            upper_zone_ratio=0.58,
            mid_zone_ratio=0.25,
            lower_zone_ratio=0.17,
        ),
    },
    "pneumothorax": {
        "label": "6. Pneumothorax",
        "audio": AudioFeatures(
            spectral_centroid=890.0,
            spectral_bandwidth=600.0,
            hf_energy_ratio=0.03,
            temporal_burstiness=0.20,
            cough_rate=1.5,
            zero_crossing_rate=0.04,
            mean_intensity_db=-42.0,
        ),
        "cxr": CXRFeatures(
            mean_opacity=0.18,
            peripheral_central_ratio=0.80,
            texture_entropy=0.35,
            texture_homogeneity=0.60,
            symmetry_ratio=0.42,
            upper_zone_ratio=0.35,
            mid_zone_ratio=0.35,
            lower_zone_ratio=0.30,
        ),
    },
}


def generate(output_dir: str = "docs/sample_outputs") -> None:
    """Run analyzers on each synthetic profile and save results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    audio_analyzer = HierarchicalPhysiologyAnalyzer()
    cxr_analyzer = HierarchicalCXRAnalyzer()

    for name, profile in PROFILES.items():
        print(f"\n{'='*60}")
        print(f"  Generating: {name} ({profile['label']})")
        print(f"{'='*60}")

        # Run analysis
        audio_result = audio_analyzer.analyze(profile["audio"])
        cxr_result = cxr_analyzer.analyze(profile["cxr"])

        result = {
            "disease_profile": name,
            "expected_label": profile["label"],
            "audio_features": asdict(profile["audio"]),
            "audio_hierarchical_analysis": audio_result,
            "cxr_features": asdict(profile["cxr"]),
            "cxr_hierarchical_analysis": cxr_result,
        }

        # Save
        outfile = out / f"{name}_analysis.json"
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  -> Saved: {outfile}")

        # Print summary
        audio_primary = audio_result.get("hierarchical_analysis", {}).get("level_3", {}).get("primary_candidate", "?")
        cxr_primary = cxr_result.get("hierarchical_analysis", {}).get("level_3", {}).get("primary_candidate", "?")
        print(f"  Audio primary: {audio_primary}")
        print(f"  CXR primary:   {cxr_primary}")
        match = "✅ AGREE" if audio_primary == cxr_primary else "⚠️  DISAGREE"
        print(f"  Cross-modal:   {match}")

    print(f"\n{'='*60}")
    print(f"  All samples saved to: {out}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample pipeline outputs")
    parser.add_argument("--output-dir", default="docs/sample_outputs", help="Output directory")
    args = parser.parse_args()
    generate(args.output_dir)

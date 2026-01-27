#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from Agent2_Image.encoders import get_image_encoder
from Agent2_Image.classifiers.image_head import load_head_from_ckpt
from Agent2_Image.utils.gradcam_utils import save_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for a CXR image")
    parser.add_argument("--image", required=True, help="Path to CXR image")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--out-dir", default=None, help="Override output dir")
    parser.add_argument("--alpha", type=float, default=None, help="Overlay alpha")
    parser.add_argument("--target", action="append", choices=["embedding", "classifier"])
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    gcfg = cfg.get("gradcam") or {}
    targets = args.target or gcfg.get("targets", ["embedding"])
    output_dir = args.out_dir or gcfg.get("output_dir", "outputs/gradcam")
    alpha = args.alpha if args.alpha is not None else float(gcfg.get("alpha", 0.4))

    use_stub = bool(cfg.get("use_stub_encoders", True))
    image_checkpoint = cfg.get("image_checkpoint")
    encoder = get_image_encoder(use_stub=use_stub, checkpoint_path=image_checkpoint)

    if "embedding" in targets:
        heatmap = encoder.gradcam(args.image, target="embedding")
        out_path = Path(output_dir) / f"{Path(args.image).stem}_embedding.png"
        save_overlay(args.image, heatmap, str(out_path), alpha=alpha)
        print(f"[gradcam] saved: {out_path}")

    if "classifier" in targets:
        icfg = cfg.get("image_classifier") or {}
        ckpt = icfg.get("checkpoint_path", "artifacts/image_classifier_head.pt")
        classifier, _ = load_head_from_ckpt(ckpt)
        heatmap = encoder.gradcam(args.image, target="classifier", classifier=classifier)
        out_path = Path(output_dir) / f"{Path(args.image).stem}_classifier.png"
        save_overlay(args.image, heatmap, str(out_path), alpha=alpha)
        print(f"[gradcam] saved: {out_path}")


if __name__ == "__main__":
    main()


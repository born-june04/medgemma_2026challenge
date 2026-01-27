import argparse
from typing import Optional

from pipeline.core import load_config, run_pipeline, run_pipeline_modes, to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal audio+image pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--audio", help="Path to audio file")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--patient-id", help="Patient identifier (requires pairs_index)")
    parser.add_argument("--pairs-index", help="CSV with patient_id,audio_path,image_path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_pipeline(
        audio_path=args.audio,
        image_path=args.image,
        config=cfg,
        patient_id=args.patient_id,
        pairs_index=args.pairs_index,
    )
    print(to_json(result))


if __name__ == "__main__":
    main()

import argparse

from pipeline import load_config, run_pipeline_modes, to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare demo pipeline CLI")
    parser.add_argument("--audio", required=True, help="Path to cough/breath .wav file")
    parser.add_argument("--image", required=True, help="Path to CXR image (.png/.jpg)")
    parser.add_argument("--intake", default="", help="Optional intake text")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = run_pipeline_modes(args.audio, args.image, args.intake, config=config)
    print(to_json(outputs))


if __name__ == "__main__":
    main()

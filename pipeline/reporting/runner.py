from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from pipeline.reporting.medgemma_report import generate_medgemma_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedGemma report generation in an isolated environment")
    parser.add_argument("--model-id", required=True, help="HF model id (e.g., google/medgemma-4b-it)")
    parser.add_argument("--inputs-json", required=True, help="Path to pipeline payload JSON")
    parser.add_argument("--out-dir", required=True, help="Output directory for prompt/report/error files")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    payload: Dict[str, Any] = json.loads(Path(args.inputs_json).read_text(encoding="utf-8"))
    result = generate_medgemma_report(
        payload,
        model_id=args.model_id,
        out_dir=args.out_dir,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()



#!/bin/bash
# Run pipeline with hierarchical physiology analysis (Audio + CXR)
# Usage: Activate your environment, then run this script from the project root.

# Activate conda environment (adjust to your setup)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

# Run pipeline with patient P0001
python pipeline.py \
  --config configs/config.yaml \
  --patient-id P0001 \
  --pairs-index data/pairs.csv

echo ""
echo "Pipeline completed! Check outputs:"
echo "  - Audio hierarchical analysis: outputs/evidence/physiology/P0001/hierarchical_analysis.json"
echo "  - Audio raw features: outputs/evidence/physiology/P0001/physiology.json"
echo "  - CXR hierarchical analysis: outputs/evidence/cxr_physiology/P0001/hierarchical_analysis.json"
echo "  - CXR raw features: outputs/evidence/cxr_physiology/P0001/cxr_features.json"
echo "  - Report: outputs/reports/P0001/report.txt"

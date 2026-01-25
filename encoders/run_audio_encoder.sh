#!/bin/bash
# Wrapper script to run audio_encoder.py with correct LD_LIBRARY_PATH

# Activate conda environment if not already activated
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_PREFIX" != *"kaggle"* ]]; then
    echo "Activating kaggle conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate kaggle
fi

# Set LD_LIBRARY_PATH to use conda environment's libstdc++
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    echo "Set LD_LIBRARY_PATH to: $LD_LIBRARY_PATH"
fi

# Run the script
cd "$(dirname "$0")"
python audio_encoder.py "$@"


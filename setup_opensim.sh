#!/bin/bash
# Flekks OpenSim Environment Setup
#
# Creates a conda environment with OpenSim, Pose2Sim, and all dependencies.
# Run once on the server or local machine to set up the OpenSim pipeline.
#
# Prerequisites: conda or miniconda installed
#   curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname -s)-$(uname -m).sh -o miniforge.sh
#   bash miniforge.sh
#
# Usage: bash setup_opensim.sh

set -euo pipefail

ENV_NAME="flekks-opensim"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

echo "=== Flekks OpenSim Environment Setup ==="
echo ""

# Step 1: Create conda environment
echo "[1/5] Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.11 numpy -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Step 2: Install OpenSim from conda
echo "[2/5] Installing OpenSim Python bindings"
conda install -c opensim-org opensim -y

# Step 3: Install pip packages
echo "[3/5] Installing Pose2Sim + dependencies"
pip install pose2sim>=0.10.0 pygltflib pandas tensorflow opencv-python mediapipe

# Step 4: Download Rajagopal model
echo "[4/5] Downloading Rajagopal2016 model"
mkdir -p "$MODELS_DIR"
if [ ! -f "$MODELS_DIR/Rajagopal2016.osim" ]; then
    curl -L "https://raw.githubusercontent.com/opensim-org/opensim-models/master/Models/Rajagopal/Rajagopal2016.osim" \
        -o "$MODELS_DIR/Rajagopal2016.osim"
    echo "  Downloaded: $MODELS_DIR/Rajagopal2016.osim"
else
    echo "  Already exists: $MODELS_DIR/Rajagopal2016.osim"
fi

# Step 5: Verify imports
echo "[5/5] Verifying installation"
python -c "
import opensim
import pose2sim
import pygltflib
import pandas
import numpy
import cv2
import mediapipe
print('All imports OK')
print(f'  OpenSim: {opensim.__version__}')
print(f'  Pose2Sim: {pose2sim.__version__}')
print(f'  NumPy: {numpy.__version__}')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate: conda activate $ENV_NAME"
echo "To run with OpenSim: python backend_processor.py video.mp4 --opensim"
echo ""
echo "Optional: Download LSTM marker augmenter weights"
echo "  python marker_augmenter.py --download"

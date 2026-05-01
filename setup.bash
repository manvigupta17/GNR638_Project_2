#!/usr/bin/env bash
# =============================================================================
# GNR638 — MCQ Visual Reasoning Pipeline: Setup Script
#
# This script is run ONCE with internet access before grading begins.
# It performs the following steps:
#   1. Creates and activates the conda environment (Python 3.11)
#   2. Installs all Python dependencies
#   3. Clones the project repository
#   4. Downloads the Qwen2.5-VL-72B-Instruct model weights
#
# Usage:
#   bash setup.bash
#
# After this script completes:
#   conda activate gnr_project_env
#   python inference.py --test_dir <absolute_path_to_test_dir>
# =============================================================================

set -e  # Exit immediately on any error

# ── Step 1: Create conda environment ─────────────────────────────────────────
echo "============================================================"
echo "Step 1: Creating conda environment: gnr_project_env (Python 3.11)"
echo "============================================================"

conda create -n gnr_project_env python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gnr_project_env

echo "Conda environment created and activated."

# ── Step 2: Clone the repository ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 2: Cloning project repository"
echo "============================================================"

# TODO: Replace with your actual public GitHub repository URL before submission.
REPO_URL="https://github.com/manvigupta17/GNR638_Project_2.git"
REPO_DIR="gnr638_project"

if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repository already cloned. Pulling latest..."
    cd "$REPO_DIR" && git pull && cd ..
fi

cd "$REPO_DIR"
echo "Working directory: $(pwd)"

# ── Step 3: Install Python dependencies ──────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 3: Installing Python dependencies from requirements.txt"
echo "============================================================"

pip install -r requirements.txt

echo "Dependencies installed."

# ── Step 4: Download model weights ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 4: Downloading Qwen2.5-VL-72B-Instruct model weights"
echo "        (~40 GB — this will take 15-30 minutes)"
echo "============================================================"

MODEL_DIR="./Qwen2.5-VL-72B-Instruct"

if [ ! -d "$MODEL_DIR" ]; then
    python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
print("Downloading Qwen/Qwen2.5-VL-72B-Instruct ...")
snapshot_download(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    local_dir="./Qwen2.5-VL-72B-Instruct",
)
print("Download complete.")
PYEOF
else
    echo "Model weights already present at $MODEL_DIR — skipping download."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Setup complete."
echo ""
echo "To run inference:"
echo "  conda activate gnr_project_env"
echo "  cd $REPO_DIR"
echo "  python inference.py --test_dir <absolute_path_to_test_dir>"
echo "============================================================"

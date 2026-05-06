#!/bin/bash
# One-time setup for the V100 virtual environment (.venv-v100).
#
# Must be run from the repo root on a node with internet access (login node):
#   bash _setup_v100/setup.sh
#
# See _setup_v100/README.md for full usage and update instructions.

set -eu
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== V100 venv setup ==="
log "Host: $(hostname)"

# Proxy needed for package downloads on Grete
source slurm_nhr/env.sh

VENV=".venv-v100"

# Step 1: create the venv (skip if already exists)
if [ ! -d "$VENV" ]; then
    log "Creating $VENV..."
    uv venv "$VENV"
else
    log "$VENV already exists, skipping creation"
fi

# Step 2: install all deps from the lock file (installs cu128 torch — overwritten next)
log "Syncing all dependencies from lock file..."
UV_PROJECT_ENVIRONMENT="$VENV" uv sync

# Step 3: overwrite torch/torchvision with cu126 builds (V100 sm_70 compatible)
log "Installing cu126 torch into $VENV..."
uv pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126 \
    --python "$VENV/bin/python" \
    --reinstall

log "=== Done ==="
log "Torch version in $VENV:"
"$VENV/bin/python" -c "import torch; print(' ', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
log "Submit V100 training with: sbatch slurm_nhr/exp01/jupyter/train_ade20k_canvit.sbatch"

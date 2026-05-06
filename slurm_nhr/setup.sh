#!/bin/bash
# One-time setup for the Grete virtual environment (.venv).
#
# Must be run from the repo root on a node with internet access (login node):
#   bash slurm_nhr/setup.sh
#
# Idempotent — safe to re-run after pyproject/uv.lock changes.

set -eu
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Grete venv setup ==="
log "Host: $(hostname)"

# Proxy needed for package downloads on Grete
source slurm_nhr/env.sh

VENV=".venv"

if [ ! -d "$VENV" ]; then
    log "Creating $VENV..."
    uv venv "$VENV"
else
    log "$VENV already exists, skipping creation"
fi

log "Syncing dependencies from lock file (cuda group)..."
UV_PROJECT_ENVIRONMENT="$VENV" uv sync

log "=== Done ==="
log "Torch version in $VENV:"
"$VENV/bin/python" -c "import torch; print(' ', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
log "Submit training with: sbatch slurm_nhr/exp01/train_ade20k_canvit.sbatch"

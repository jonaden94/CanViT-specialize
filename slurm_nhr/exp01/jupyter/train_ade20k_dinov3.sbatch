#!/bin/bash
#SBATCH -p jupyter
#SBATCH -G V100:1
#SBATCH -t 0-12:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub + Comet/W&B uploads
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --delay-boot=1
#SBATCH -o logs/exp01-ade20k-dinov3-v100/job-%j.log
#SBATCH -e logs/exp01-ade20k-dinov3-v100/job-%j.log
#SBATCH --job-name=ade20k-dinov3-v100
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# DINOv3 baseline probe training on V100 (jupyter partition).
# Requires .venv-v100 (cu126 torch for V100 sm_70 compat):
#   bash _setup_v100/setup.sh
#
# Submit with:
#   sbatch slurm_nhr/exp01/jupyter/train_ade20k_dinov3.sbatch --scene-size 512
#   sbatch slurm_nhr/exp01/jupyter/train_ade20k_dinov3.sbatch --scene-size 256 --max-steps 4000

cd /user/henrich1/u25995/jonathan/repos/CanViT-specialize
source /user/henrich1/u25995/.bashrc

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, COMET_API_KEY, WANDB_API_KEY) into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== exp01 ADE20K DINOv3 baseline probe training (V100, jupyter) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh
mkdir -p logs/exp01-ade20k-dinov3-v100

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

log "Starting training..."
exec "$VENV/bin/python" -m canvit_specialize.training.ade20k train-dinov3-probe "$@"

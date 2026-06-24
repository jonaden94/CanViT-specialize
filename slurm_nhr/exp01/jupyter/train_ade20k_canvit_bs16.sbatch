#!/bin/bash
#SBATCH -p jupyter
#SBATCH -G V100:1
#SBATCH -t 0-12:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub + Comet/W&B uploads
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --delay-boot=1
#SBATCH -o logs/exp01-ade20k-canvit-v100-bs16/job-%j.log
#SBATCH -e logs/exp01-ade20k-canvit-v100-bs16/job-%j.log
#SBATCH --job-name=ade20k-canvit-v100-bs16
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# ADE20K CanViT probe training on V100 (jupyter partition), batch_size=16 (default).
# peak_lr=3e-4 (config default — linear-scaling baseline).
#
# Requires .venv-cu126 (cu126 torch for V100 sm_70 compat):
#   UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126
#
# Submit:
#   sbatch slurm_nhr/exp01/jupyter/train_ade20k_canvit_bs16.sbatch
#   sbatch slurm_nhr/exp01/jupyter/train_ade20k_canvit_bs16.sbatch --finetune
#   sbatch slurm_nhr/exp01/jupyter/train_ade20k_canvit_bs16.sbatch --finetune \
#       --init-probe-repo canvit/probe-ade20k-40k-s512-c64-in21k     # LP-FT

cd /user/henrich1/u25995/jonathan/repos/CanViT-specialize
source /user/henrich1/u25995/.bashrc

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, COMET_API_KEY, WANDB_API_KEY) into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== exp01 ADE20K CanViT probe training (V100, jupyter, bs=16) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh
mkdir -p logs/exp01-ade20k-canvit-v100-bs16

VENV=".venv-cu126"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126"
    exit 1
fi

log "Starting training (bs=16, lr=3e-4)..."
exec "$VENV/bin/python" -m canvit_specialize.training.ade20k train \
    --batch-size 16 \
    --peak-lr 3e-4 \
    "$@"

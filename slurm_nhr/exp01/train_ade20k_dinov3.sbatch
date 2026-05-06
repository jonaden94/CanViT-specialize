#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-01:30:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub + Comet
#SBATCH --mem=32G
#SBATCH -o logs/exp01-ade20k-dinov3/job-%j.log
#SBATCH -e logs/exp01-ade20k-dinov3/job-%j.log
#SBATCH --job-name=ade20k-dinov3
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# DINOv3 baseline probe training at a given resolution.
#
# Submit with:
#   sbatch slurm_nhr/exp01/train_ade20k_dinov3.sbatch --scene-size 512
#   sbatch slurm_nhr/exp01/train_ade20k_dinov3.sbatch --scene-size 256 --teacher-repo facebook/dinov3-vits16-pretrain-lvd1689m

cd /user/henrich1/u25995/jonathan/repos/CanViT-specialize
source /user/henrich1/u25995/.bashrc

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, COMET_API_KEY) into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== exp01 ADE20K DINOv3 baseline probe training (A100, grete:shared) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh
mkdir -p logs/exp01-ade20k-dinov3

VENV=".venv"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash slurm_nhr/setup.sh"
    exit 1
fi

log "Starting training..."
exec "$VENV/bin/python" -m canvit_specialize.training.ade20k train-dinov3-probe "$@"

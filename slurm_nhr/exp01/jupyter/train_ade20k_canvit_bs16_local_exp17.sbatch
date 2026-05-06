#!/bin/bash
#SBATCH -p jupyter
#SBATCH -G V100:1
#SBATCH -t 0-12:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub + Comet/W&B uploads
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --delay-boot=1
#SBATCH -o logs/exp01-ade20k-canvit-v100-bs16-local-exp17/job-%j.log
#SBATCH -e logs/exp01-ade20k-canvit-v100-bs16-local-exp17/job-%j.log
#SBATCH --job-name=ade20k-canvit-v100-bs16-local-exp17
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# ADE20K CanViT probe training on V100 (jupyter), batch_size=16 (default lr).
# Identical to train_ade20k_canvit_bs16.sbatch except --model-repo points at a
# locally converted CanViT-pretrain checkpoint (exp17-1gpu/step-458752) instead
# of the default HuggingFace-Hub one.
#
# The local dir was produced by:
#   .venv/bin/python scripts/pretrain_ckpt_to_hf_format.py \
#       --pt-path /user/henrich1/u25995/jonathan/repos/CanViT-pretrain/checkpoints/exp17-1gpu/step-458752.pt \
#       --out-dir checkpoints/local-canvit/exp17-1gpu-step-458752
#
# Submit:
#   sbatch slurm_nhr/exp01/jupyter/train_ade20k_canvit_bs16_local_exp17.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-specialize
source /user/henrich1/u25995/.bashrc

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, COMET_API_KEY, WANDB_API_KEY) into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== exp01 ADE20K CanViT probe training (V100, jupyter, bs=16, local exp17 ckpt) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh
mkdir -p logs/exp01-ade20k-canvit-v100-bs16-local-exp17

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

LOCAL_CKPT_DIR="/user/henrich1/u25995/jonathan/repos/CanViT-specialize/checkpoints/local-canvit/exp17-1gpu-step-458752"
if [ ! -f "$LOCAL_CKPT_DIR/model.safetensors" ]; then
    log "ERROR: converted checkpoint missing at $LOCAL_CKPT_DIR"
    log "       run: .venv/bin/python scripts/pretrain_ckpt_to_hf_format.py --pt-path <...> --out-dir <...>"
    exit 1
fi

log "Starting training (bs=16, lr=3e-4, model=$LOCAL_CKPT_DIR)..."
exec "$VENV/bin/python" -m canvit_specialize.training.ade20k train \
    --batch-size 16 \
    --peak-lr 3e-4 \
    --model-repo "$LOCAL_CKPT_DIR" \
    "$@"

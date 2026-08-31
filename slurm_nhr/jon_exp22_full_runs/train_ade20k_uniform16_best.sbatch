#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-08:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # HuggingFace Hub (DINOv3 teacher for feature dims)
#SBATCH --mem=64G
#SBATCH -o logs/jon_exp22_full_runs/ade20k-uniform16-best/job-%j.log
#SBATCH -e logs/jon_exp22_full_runs/ade20k-uniform16-best/job-%j.log
#SBATCH --job-name=ade20k-exp22-uniform16

# ADE20K FROZEN-PROBE training on the exp22-uniform16 BEST checkpoint
# (step-1515520, the max of val/scene_cos_norm_t9 over the run — the same
# checkpoint being evaluated for IN1k/reconstruction under jon_exp22_full_runs).
# Only the SegmentationProbe head trains; the CanViT backbone is frozen
# (finetune=False default -> model.requires_grad_(False) in train_canvit.py).
#
# Loads the LOCAL HF-format checkpoint dir directly (same layout eval uses):
#   .../exp22-uniform16/checkpoints/step-1515520-hf
# Protocol = ADE20K probe defaults: random viewpoints, n_timesteps=10,
# scene 512px, canvas_grid=512//16=32 (matches the eval grid), 40k steps.
#
# NOTE: uses .venv-cu126 (the repo's only venv; the old exp01 launcher's
# hardcoded ".venv" does not exist here).
#
# Submit:  sbatch slurm_nhr/jon_exp22_full_runs/train_ade20k_uniform16_best.sbatch
#   (extra CLI args are forwarded, e.g. --max-steps 20000)

cd /user/henrich1/u25995/jonathan/repos/CanViT-specialize
source /user/henrich1/u25995/.bashrc

set -eu  # NOT -x: would trace secret exports (HF_TOKEN, COMET_API_KEY) into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== jon_exp22_full_runs :: ADE20K CanViT frozen-probe :: exp22-uniform16 best (step-1515520) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh
mkdir -p logs/jon_exp22_full_runs/ade20k-uniform16-best

VENV=".venv-cu126"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash slurm_nhr/setup.sh"
    exit 1
fi

MODEL_REPO="/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/jon_exp22_full_runs/exp22-uniform16/checkpoints/step-1515520-hf"
[ -d "$MODEL_REPO" ] || { log "ERROR: HF checkpoint missing: $MODEL_REPO"; exit 1; }

log "Starting ADE20K probe training on: $MODEL_REPO"
exec "$VENV/bin/python" -m canvit_specialize.training.ade20k train \
    --model-repo "$MODEL_REPO" \
    --wandb-project jon_exp22_full_runs \
    "$@"

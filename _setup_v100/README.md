# V100 Environment Setup

Two virtual environments live side by side:

| venv          | torch build | target hardware | managed by                            |
|---------------|-------------|-----------------|---------------------------------------|
| `.venv`       | cu128       | A100 / H100     | `slurm_nhr/setup.sh` (uv sync)        |
| `.venv-v100`  | cu126       | V100 (sm_70)    | `_setup_v100/setup.sh` (sync + override) |

The only difference is the torch / torchvision wheel. Everything else comes from the
same `uv.lock`.

## First-time setup

Run once from the repo root on a login node (has internet via GWDG proxy):

```bash
bash _setup_v100/setup.sh
```

This:
1. Creates `.venv-v100`
2. Installs all dependencies from `uv.lock`
3. Overwrites torch / torchvision with cu126 builds

## Submitting V100 training

```bash
sbatch slurm_nhr/exp01/jupyter/train_ade20k_canvit.sbatch
sbatch slurm_nhr/exp01/jupyter/train_ade20k_dinov3.sbatch --scene-size 512
```

## After updating dependencies (new package added to `pyproject.toml`)

Re-run the setup script — it is idempotent:

```bash
bash _setup_v100/setup.sh
```

This re-syncs all deps from the updated lock file, then re-overwrites torch with cu126.

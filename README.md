# CanViT-specialize

Training loops for [CanViT](https://github.com/m2b3/CanViT-PyTorch) downstream probes (ADE20K segmentation) and IN1k finetuning.

## Install

To **use** this package elsewhere:

```bash
uv add "canvit-specialize @ git+https://github.com/m2b3/CanViT-specialize.git"
```

For TPU finetuning, see [`gcp_in1k_clf_ft/README.md`](canvit_specialize/training/gcp_in1k_clf_ft/README.md).

### Local multi-repo setup (development)

This repo is part of a five-repo active-vision project that is developed
together, with `CanViT-specialize` depending on `CanViT-PyTorch`:

```
repos/
├── fovi/               # leaf — no internal deps
├── CanViT-PyTorch/     # depends on fovi
├── CanViT-specialize/  # this repo; depends on CanViT-PyTorch
├── CanViT-pretrain/    # depends on CanViT-PyTorch[fovi]
└── CanViT-eval/        # depends on CanViT-PyTorch[fovi] + CanViT-specialize
```

Each repo has its **own** uv-managed venv. Clone all five **as siblings in the
same parent folder**, then:

```bash
# Default env (.venv) — H100 (sm_90)
uv sync

# V100 + A100 env (.venv-cu126) — cu126 torch (Grete V100 + A100 partitions)
UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126
```

The two envs are conflicting, separately-locked resolutions: torch is pinned in
the `cuda` (default) and `cu126` dependency groups in `pyproject.toml`, so each
`uv sync` is reproducible. cu126 wheels keep the sm_70 (V100) support the default
cu128 wheels dropped. (`CanViT-specialize` also has a TPU/cpu `gcp-in1k-finetune`
group — see `pyproject.toml`.)

The cross-repo link is committed in `pyproject.toml` under `[tool.uv.sources]`
as a **relative-path editable install**
(`canvit-pytorch = { path = "../CanViT-PyTorch", editable = true }`). Relative
paths resolve on any machine as long as the repos are siblings, and the editable
install means edits in the local `CanViT-PyTorch` clone (and, transitively,
`fovi`) are picked up immediately — no reinstall, no manual `uv pip install -e`.
To install without the sibling clones, swap that line back to the remote fork
(`canvit-pytorch = { git = "https://github.com/jonaden94/CanViT-PyTorch.git" }`)
and `uv sync`.

> For frozen multi-day runs, see `CanViT-pretrain`'s README
> ("Pinning code for long runs").

## Using a pre-trained probe

```python
from canvit_pytorch import SegmentationProbe
probe = SegmentationProbe.from_pretrained("canvit/probe-ade20k-40k-s512-c64-in21k")
logits = probe(features)  # [B, H, W, D] → [B, num_classes, H, W]
```

For the fused **CanViT + probe** pair, see `canvit_pytorch.CanViTForSemanticSegmentation`.

## Training

`ADE20K_ROOT` must be set before training, plus credentials for whichever
experiment tracker you pick (W&B is the default — set `WANDB_PROJECT`; or
pass `--tracker comet` and set `COMET_API_KEY` + `COMET_WORKSPACE`; or
`--tracker none` to disable logging).

```bash
cp .envrc.example .envrc && direnv allow
# Edit .envrc to point at your dataset and tracker config.
```

### ADE20K segmentation probe (frozen CanViT)

```bash
uv run python -m canvit_specialize.training.ade20k train \
  --scene-size 1024 --canvas-grid 64
```

### DINOv3 baseline probe

```bash
uv run python -m canvit_specialize.training.ade20k train-dinov3-probe
```

### IN1k classification finetuning on GCP TPU v6e

See [`canvit_specialize/training/gcp_in1k_clf_ft/README.md`](canvit_specialize/training/gcp_in1k_clf_ft/README.md).

## Citation

```bibtex
@article{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.

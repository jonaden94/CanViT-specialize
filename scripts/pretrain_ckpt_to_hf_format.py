"""Convert a CanViT-pretrain `.pt` checkpoint into the local HF Hub layout
that `CanViTForPretrainingHFHub.from_pretrained(<dir>)` reads.

Output dir gets:
    config.json        — backbone_name, model_config, canvas_patch_grid_sizes
    model.safetensors  — model state_dict

Usage:
    .venv/bin/python scripts/pretrain_ckpt_to_hf_format.py \\
        --pt-path  /path/to/step-NNNNNN.pt \\
        --out-dir  /path/to/output/dir
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Args:
    pt_path: Path
    out_dir: Path


def _migrate_standardizers_in_place(raw: dict) -> None:
    """Mirror canvit_pretrain's legacy → current standardizer migration."""
    if (scene_legacy := raw.get("scene_norm_state")) is None:
        return
    cls_legacy = raw["cls_norm_state"]
    grids = raw["canvas_patch_grid_sizes"]
    assert len(grids) == 1, f"Expected single grid size, got {grids}"
    G = str(grids[0])
    sd = raw["state_dict"]
    for prefix, legacy in [("scene_standardizers", scene_legacy), ("cls_standardizers", cls_legacy)]:
        for stat in ("mean", "var", "_initialized"):
            sd[f"{prefix}.{G}.{stat}"] = legacy[stat]
    del raw["scene_norm_state"], raw["cls_norm_state"]
    log.info("Migrated legacy standardizers (grid=%s)", G)


def main(args: Args) -> None:
    log.info("Loading %s ...", args.pt_path)
    raw = torch.load(args.pt_path, map_location="cpu", weights_only=False)
    _migrate_standardizers_in_place(raw)

    config = {
        "backbone_name": raw["backbone_name"],
        "model_config": raw["model_config"],
        "canvas_patch_grid_sizes": raw["canvas_patch_grid_sizes"],
        "metadata": {
            "source_pt": str(args.pt_path),
            "step": raw.get("step"),
            "teacher_name": raw.get("teacher_name"),
            "dataset": raw.get("dataset"),
            "timestamp": raw.get("timestamp"),
            "git_commit": raw.get("git_commit"),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out_dir / "config.json"
    sd_path = args.out_dir / "model.safetensors"
    cfg_path.write_text(json.dumps(config, indent=2, default=str))
    save_file(raw["state_dict"], sd_path)

    log.info("Wrote %s (%d params)", sd_path, len(raw["state_dict"]))
    log.info("Wrote %s", cfg_path)
    log.info("Load with: CanViTForPretrainingHFHub.from_pretrained(%r)", str(args.out_dir))


if __name__ == "__main__":
    main(tyro.cli(Args))

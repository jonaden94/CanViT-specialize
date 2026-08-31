"""Convert an ADE20K SegmentationProbe training `.pt` into the local HF layout
that `SegmentationProbe.from_pretrained(<dir>)` (and the CanViT-eval
`ade20k-seg-canvit` task) reads — WITHOUT touching the HuggingFace Hub.

This is the segmentation analog of `pretrain_ckpt_to_hf_format.py` (which does
the same for the CanViT backbone). `push_probes.py` also produces this format,
but only by uploading to the Hub; this script writes it to local disk so the
train (canvit-specialize) -> eval (canvit-eval) loop needs no network and no
Hub account.

Output dir gets (written by PyTorchModelHubMixin.save_pretrained):
    config.json        — probe __init__ kwargs (embed_dim, num_classes, dropout, use_ln)
    model.safetensors  — probe head state_dict

Usage:
    .venv-cu126/bin/python scripts/probe_ckpt_to_hf_format.py \\
        --pt-path  /path/to/canvas_hidden_best_t9_miou0.XXXX_step<N>.pt \\
        --out-dir  /path/to/output/dir

Load with (no network):
    SegmentationProbe.from_pretrained("/path/to/output/dir")
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from canvit_pytorch.probes import SegmentationProbe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Args:
    pt_path: Path
    out_dir: Path


def main(args: Args) -> None:
    log.info("Loading %s ...", args.pt_path)
    raw = torch.load(args.pt_path, map_location="cpu", weights_only=False)

    # Refuse LP-FT checkpoints: they carry full CanViT weights alongside the
    # probe head; converting only the head would silently drop the backbone.
    if "model_state_dict" in raw or raw.get("config", {}).get("finetune") is True:
        raise NotImplementedError(
            f"{args.pt_path.name} is an LP-FT checkpoint with full CanViT weights. "
            "This converter only handles standalone frozen-probe heads."
        )

    if "probe_state_dict" not in raw:
        raise AssertionError(f"Not a probe checkpoint. Keys: {sorted(raw.keys())}")
    sd = raw["probe_state_dict"]

    # Reconstruct the probe geometry from the state dict + saved config, exactly
    # as push_probes.load_probe does, so the two paths produce identical repos.
    embed_dim = sd["conv.weight"].shape[1]
    num_classes = sd["conv.weight"].shape[0]
    use_ln = "ln.weight" in sd
    dropout = raw.get("config", {}).get("dropout")
    assert dropout is not None, f"dropout missing in {args.pt_path}"

    probe = SegmentationProbe(
        embed_dim=embed_dim, num_classes=num_classes, dropout=dropout, use_ln=use_ln,
    )
    result = probe.load_state_dict(sd, strict=True)
    assert not result.missing_keys and not result.unexpected_keys, (
        f"state_dict mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
    )
    log.info(
        "Probe: embed_dim=%d num_classes=%d use_ln=%s dropout=%s (step=%s, feat=%s)",
        embed_dim, num_classes, use_ln, dropout, raw.get("step"), raw.get("feat_type"),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probe.save_pretrained(args.out_dir)  # writes config.json + model.safetensors, no network
    log.info("Wrote %s/{config.json,model.safetensors}", args.out_dir)
    log.info("Load with: SegmentationProbe.from_pretrained(%r)", str(args.out_dir))


if __name__ == "__main__":
    main(tyro.cli(Args))

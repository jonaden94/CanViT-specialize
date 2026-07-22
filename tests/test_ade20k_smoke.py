"""CPU smoke tests for the ADE20K probe training path (unification-status §5.8).

Exercises the exact per-step code path of train_canvit.py on a tiny CPU model:
random viewpoints -> extract_canvas_features rollout (frozen backbone, glimpse
pre-crop, uniform patcher, glimpse_size_px=None load path) -> SegmentationProbe
per timestep -> ce_loss -> backward. A test like this would have caught the
2026-07 `glimpse=` kwarg breakage (unification-status §5.2) in seconds instead
of three months.
"""

from types import SimpleNamespace

import pytest
import torch
from canvit_pytorch import CanViTForPretrainingHFHub
from canvit_pytorch.probes import SegmentationProbe

from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
from canvit_specialize.training.ade20k.features import extract_canvas_features
from canvit_specialize.training.ade20k.loss import ce_loss
from canvit_specialize.training.utils import make_viewpoints

_B, _G, _T, _IMG = 2, 8, 2, 224


@pytest.fixture()
def model() -> CanViTForPretrainingHFHub:
    torch.manual_seed(0)
    m = CanViTForPretrainingHFHub(
        backbone_name="vits16",
        model_config={"teacher_dim": 384},
        canvas_patch_grid_sizes=[_G],
    )
    # the trainer freezes the backbone (train_canvit.py) — mirror that here
    m.requires_grad_(False)
    m.eval()
    return m


def test_two_step_rollout_trains_probe_head_only(model: CanViTForPretrainingHFHub) -> None:
    torch.manual_seed(1)
    images = torch.randn(_B, 3, _IMG, _IMG)
    masks = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG))
    masks[:, :4] = IGNORE_LABEL  # exercise the ignore-index path
    vps = make_viewpoints(
        "random", _B, torch.device("cpu"), _T,
        min_scale=0.3, max_scale=1.0, start_with_full_scene=True,
    )
    assert len(vps) == _T

    feats = extract_canvas_features(
        model=model, images=images, canvas_grid=_G, glimpse_px=128, viewpoints=vps,
    )
    hidden0 = feats.get("canvas_hidden", 0)
    assert hidden0.shape[:3] == (_B, _G, _G)
    assert torch.isfinite(hidden0).all()

    head = SegmentationProbe(hidden0.shape[-1], NUM_CLASSES)
    logits = [head(feats.get("canvas_hidden", t).float()) for t in range(_T)]
    assert logits[0].shape == (_B, NUM_CLASSES, _G, _G)
    loss = torch.stack([ce_loss(lg, masks) for lg in logits]).mean()
    assert torch.isfinite(loss)
    loss.backward()

    head_grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert head_grads and any(g.abs().sum() > 0 for g in head_grads)
    # frozen backbone stays untouched
    assert all(p.grad is None for p in model.parameters())


def test_foveated_model_rejected_loudly() -> None:
    """The double-crop guard: pre-cropping is wrong for foveated/square patchers,
    so extract_canvas_features must fail fast instead of silently double-cropping."""
    fake = SimpleNamespace(cfg=SimpleNamespace(patcher_name="foveated"))
    with pytest.raises(AssertionError, match="uniform"):
        extract_canvas_features(
            model=fake,  # type: ignore[arg-type]  # guard fires before any model use
            images=torch.zeros(1, 3, 64, 64),
            canvas_grid=_G,
            glimpse_px=128,
            viewpoints=[],
        )

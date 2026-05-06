"""Experiment tracker abstraction.

Wraps an optional Comet experiment OR an optional wandb run behind a single
interface that mirrors the surface used by `train_canvit.py`, `train_dinov3.py`,
and `viz.py`. `make_tracker` selects the backend at job start based on
`cfg.tracker`. When neither backend is active (`cfg.tracker == "none"`),
every method is a no-op so call sites stay unconditional.
"""

from __future__ import annotations

import gc
import io
import logging
from pathlib import Path
from typing import Any

import comet_ml
import matplotlib.pyplot as plt
import wandb

log = logging.getLogger(__name__)


class Tracker:
    """Fans `log_*` calls out to whichever backend is active.

    `comet_exp` and `wandb_run` are mutually exclusive in practice (the
    selector exposes "comet", "wandb", or "none"), but both being None is
    a valid no-op.
    """

    def __init__(
        self,
        comet_exp: comet_ml.CometExperiment | None = None,
        wandb_run: Any | None = None,
    ) -> None:
        self._comet = comet_exp
        self._wandb = wandb_run

    def log_parameters(self, params: dict[str, Any]) -> None:
        if self._comet is not None:
            self._comet.log_parameters(params)
        if self._wandb is not None:
            self._wandb.config.update(params, allow_val_change=True)

    def log_metric(self, name: str, value: Any, step: int | None = None) -> None:
        if self._comet is not None:
            self._comet.log_metric(name, value, step=step)
        if self._wandb is not None:
            self._wandb.log({name: value}, step=step)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._comet is not None:
            self._comet.log_metrics(metrics, step=step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_curve(self, name: str, *, x: list, y: list, step: int | None = None) -> None:
        if self._comet is not None:
            self._comet.log_curve(name, x=x, y=y, step=step)
        if self._wandb is not None:
            # wandb has no native log_curve — render as image, same as PCA figures.
            img = _curve_as_pil(name, x, y)
            self._wandb.log({name: wandb.Image(img)}, step=step)

    def log_image(self, image: Any, name: str, step: int | None = None) -> None:
        if self._comet is not None:
            self._comet.log_image(image, name=name, step=step)
        if self._wandb is not None:
            self._wandb.log({name: wandb.Image(image)}, step=step)

    def add_tag(self, tag: str) -> None:
        if self._comet is not None:
            self._comet.add_tag(tag)
        if self._wandb is not None:
            # wandb stores tags as a tuple on the run object; assign a new tuple.
            self._wandb.tags = tuple(self._wandb.tags or ()) + (tag,)

    def get_comet_id(self) -> str | None:
        return self._comet.get_key() if self._comet is not None else None

    def get_wandb_id(self) -> str | None:
        return self._wandb.id if self._wandb is not None else None

    def get_key(self) -> str:
        """Stable identifier for this run. Prefers wandb, falls back to comet."""
        return self.get_wandb_id() or self.get_comet_id() or "no-tracker"

    def end(self) -> None:
        if self._comet is not None:
            self._comet.end()
        if self._wandb is not None:
            self._wandb.finish()


def _curve_as_pil(name: str, x: list, y: list) -> Any:
    """Render a line plot as a PIL.Image so wandb.Image can wrap it."""
    from PIL import Image as PILImage

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("timestep")
    ax.set_ylabel(name)
    ax.set_title(name)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    img = PILImage.open(buf)
    img.load()  # force decode before buf goes out of scope
    return img


def make_tracker(
    *,
    tracker: str,
    run_name: str,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_dir: Path | None,
) -> Tracker:
    """Build the tracker for this job."""
    if tracker == "none":
        return Tracker()

    if tracker == "comet":
        comet_cfg = comet_ml.ExperimentConfig(auto_metric_logging=False, name=run_name)
        log.info("Creating NEW Comet experiment")
        exp = comet_ml.start(experiment_config=comet_cfg)
        return Tracker(comet_exp=exp)

    if tracker == "wandb":
        assert wandb_project, "tracker='wandb' requires --wandb-project (or set WANDB_PROJECT in env)"
        if wandb_dir is not None:
            wandb_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "project": wandb_project,
            "name": run_name,
        }
        if wandb_dir is not None:
            kwargs["dir"] = str(wandb_dir)
        if wandb_entity:
            kwargs["entity"] = wandb_entity
        log.info("Creating NEW wandb run")
        run = wandb.init(**kwargs)
        return Tracker(wandb_run=run)

    raise ValueError(f"Unknown tracker: {tracker!r} (expected 'comet', 'wandb', or 'none')")

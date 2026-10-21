"""Train the in-house models of the registry: PatchTST checkpoints and Chronos-Bolt LoRA adapters."""

from __future__ import annotations

from tsfm_arena.data.panel import ReturnPanel
from tsfm_arena.logging import get_logger
from tsfm_arena.models import chronos
from tsfm_arena.models.factory import ModelFactory
from tsfm_arena.models.patchtst import load_checkpoint, save_checkpoint
from tsfm_arena.models.spec import ModelSpec, Registry
from tsfm_arena.training.lora_tune import lora_rank_tune
from tsfm_arena.training.trainer import train_patchtst

log = get_logger(__name__)


def train_spec(spec: ModelSpec, registry: Registry, factory: ModelFactory, panel: ReturnPanel) -> None:
    """Train ``spec`` on ``panel`` up to its declared cutoff and save the checkpoint.

    A ``patchtst`` child with ``init_from`` starts from the parent's checkpoint, which must exist (use
    :meth:`Registry.training_order`). A ``chronos`` spec with a path trains a LoRA adapter on its base.
    """
    if spec.training is None or spec.pretraining_cutoff is None:
        raise ValueError(f"{spec.name} has no training section or no declared cutoff")
    settings = factory.settings
    ev = settings.evaluation
    if spec.adapter == "patchtst":
        init = None
        if spec.training.init_from:
            parent = registry.get(spec.training.init_from)
            init, _ = load_checkpoint(factory.checkpoint_path(parent))
        result = train_patchtst(
            panel,
            spec.pretraining_cutoff,
            spec.training,
            settings.training,
            ev.horizon,
            list(ev.quantiles),
            settings.seed,
            init,
        )
        path = save_checkpoint(factory.checkpoint_path(spec), result.net, result.meta)
        log.info("trained_patchtst", model=spec.name, path=str(path), final_loss=result.meta["final_loss"])
        return
    if spec.adapter == "chronos":
        pipeline = factory.load_chronos_pipeline(spec)
        model = chronos.add_lora(pipeline.model, settings.training.lora)
        context = min(int(spec.params["context_length"]), panel.n_periods // 2)
        meta = lora_rank_tune(
            model,
            panel,
            spec.pretraining_cutoff,
            spec.training,
            settings.training,
            ev.horizon,
            context,
            settings.seed,
        )
        path = chronos.save_lora(factory.checkpoint_path(spec), model, meta)
        log.info("trained_lora", model=spec.name, path=str(path))
        return
    raise ValueError(f"adapter {spec.adapter!r} has no training procedure")

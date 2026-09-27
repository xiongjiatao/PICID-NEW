"""Conditional final-test execution used by the command-line runner."""

import logging
from typing import Any

from lightning import LightningDataModule, Trainer
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def run_test_stage(
    cfg: DictConfig,
    trainer: Trainer,
    model: Any,
    datamodule: LightningDataModule,
) -> list[dict[str, Any]] | None:
    """Run test only when enabled; preserve both checkpoint and one-shot paths."""
    if not cfg.get("test", True):
        logger.info("Skipping final test evaluation because test=false.")
        return None

    if cfg.trainer.max_epochs > 1:
        return trainer.test(ckpt_path="best", datamodule=datamodule)

    logger.info("Starting testing without training (used for pre-trained models)!")
    return trainer.test(model=model, datamodule=datamodule)

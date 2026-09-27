"""Protect validation-only runs from accessing held-out test data."""

from unittest.mock import Mock

from omegaconf import OmegaConf

from picid.run_test_stage import run_test_stage


def test_test_false_skips_final_test_evaluation():
    cfg = OmegaConf.create({"test": False, "trainer": {"max_epochs": 1}})
    trainer = Mock()

    result = run_test_stage(
        cfg=cfg,
        trainer=trainer,
        model=object(),
        datamodule=object(),
    )

    assert result is None
    trainer.test.assert_not_called()


def test_default_test_true_keeps_single_epoch_fit_predict_behavior():
    cfg = OmegaConf.create({"trainer": {"max_epochs": 1}})
    trainer = Mock()
    expected = [{"test/loss": 0.25}]
    trainer.test.return_value = expected
    model = object()
    datamodule = object()

    result = run_test_stage(
        cfg=cfg,
        trainer=trainer,
        model=model,
        datamodule=datamodule,
    )

    assert result == expected
    trainer.test.assert_called_once_with(model=model, datamodule=datamodule)

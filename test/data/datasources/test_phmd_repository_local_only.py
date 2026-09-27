"""PHMD local-only access must not invoke external download code."""

import pytest

from picid.data.datasources.base import phmd_repository
from picid.data.datasources.base.phmd_repository import PHMDRepository


def test_local_only_uses_preextracted_dataset_without_download(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    expected = cache / "datasets" / "XJTU-SY" / "train"
    expected.mkdir(parents=True)

    class FakeTask:
        meta = {"task": "rul"}

        def __getitem__(self, fold):
            return {"fold": fold}

    class FakeDataset:
        def __init__(self, data_name, cache_dir):
            self.data_name = data_name
            self.cache_dir = cache_dir

        def get_task(self, task_name):
            phmd_repository.datasets.download(self.data_name, cache_dir=self.cache_dir)
            return FakeTask()

    def forbidden_download(*_args, **_kwargs):
        raise AssertionError("local-only mode attempted a network download")

    monkeypatch.setattr(phmd_repository.datasets, "Dataset", FakeDataset)
    monkeypatch.setattr(phmd_repository.datasets, "download", forbidden_download)
    repository = PHMDRepository(
        data_name="XJTU-SY",
        cache_dir=str(cache),
        fold=0,
        task_mode="rul",
        download_policy="local_only",
    )

    bundle = repository.load_task_bundle(["rul"])

    assert bundle.task_folds == {"rul": {"fold": 0}}
    assert bundle.meta_data == {"task": "rul"}


def test_local_only_fails_with_missing_files_without_download(tmp_path, monkeypatch):
    def forbidden_download(*_args, **_kwargs):
        raise AssertionError("local-only mode attempted a network download")

    monkeypatch.setattr(phmd_repository.datasets, "download", forbidden_download)
    repository = PHMDRepository(
        data_name="XJTU-SY",
        cache_dir=str(tmp_path / "empty-cache"),
        fold=0,
        task_mode="rul",
        download_policy="local_only",
    )

    with pytest.raises(FileNotFoundError, match="no network download was attempted"):
        repository.load_task_bundle(["rul"])

import os

from picid.research.monitoring import (
    eta_seconds,
    gpu_memory_for_process,
    process_tree_stats,
    progress_from_log,
    progress_rate,
)


def test_gpu_memory_sampler_attributes_only_the_tracked_process():
    gpu_rows = "0, GPU-a, 20000, 4000, 80\n3, GPU-d, 24000, 500, 5"
    process_rows = "123, GPU-a, 4000\n456, GPU-a, 700\n123, GPU-d, 250"

    assert gpu_memory_for_process(gpu_rows, process_rows, 123) == {0: 4000, 3: 250}


def test_progress_parser_reads_latest_ensemble_and_epoch_markers(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("ensembles: 25%|██▌ | 2/8\nEpoch 3: 40%|████ | 20/50\n")

    progress = progress_from_log(log)

    assert progress == {"stage": "ensemble", "key": "ensemble", "completed": 2, "total": 8}
    assert eta_seconds(progress, {"key": "ensemble", "completed": 1, "monotonic": 10.0}, now=12.0) == 12.0
    assert progress_rate(progress, {"key": "ensemble", "completed": 1, "monotonic": 10.0}, now=12.0) == 0.5


def test_epoch_progress_is_monotonic_across_epoch_boundaries(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("Epoch 3: 40%|████ | 20/50\n")

    progress = progress_from_log(log, max_epochs=200)

    assert progress["key"] == "training_epochs"
    assert progress["completed"] == 170
    assert progress["total"] == 10000


def test_process_tree_sampler_reports_current_process_memory():
    result = process_tree_stats(os.getpid())

    assert result["process_count"] >= 1
    assert result["rss_mib"] > 0
    assert result["cpu_time_seconds"] >= 0

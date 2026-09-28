import os

from picid.research.monitoring import eta_seconds, process_tree_stats, progress_from_log


def test_progress_parser_reads_latest_ensemble_and_epoch_markers(tmp_path):
    log = tmp_path / "stdout.log"
    log.write_text("ensembles: 25%|██▌ | 2/8\nEpoch 3: 40%|████ | 20/50\n")

    progress = progress_from_log(log)

    assert progress == {"stage": "ensemble", "key": "ensemble", "completed": 2, "total": 8}
    assert eta_seconds(progress, {"key": "ensemble", "completed": 1, "monotonic": 10.0}, now=12.0) == 12.0


def test_process_tree_sampler_reports_current_process_memory():
    result = process_tree_stats(os.getpid())

    assert result["process_count"] >= 1
    assert result["rss_mib"] > 0
    assert result["cpu_time_seconds"] >= 0

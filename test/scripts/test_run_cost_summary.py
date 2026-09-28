from scripts.research.summarize_run_cost import summarize_run_cost


def test_gpu_memory_is_attributed_to_the_tracked_process_only():
    manifest = {
        "pid": 123,
        "seed": 72,
        "stage": "test",
        "status": "process_success_requires_result_audit",
        "exit_code": 0,
        "elapsed_seconds": 20.0,
        "physical_gpus": [0],
        "logical_device": "cuda:0",
        "cuda_visible_devices": "0",
        "initial_gpu_snapshot": {
            "concurrent_process_counts_by_physical_gpu": {"0": 2}
        },
        "shell_command": "python run.py",
        "output": "/tmp/run",
        "source_snapshot": {"sha256": "source-hash"},
    }
    telemetry = [{
        "process_tree": {"rss_mib": 300, "sum_vm_hwm_mib": 320},
        "gpus": "0, GPU-0, 20000, 1000, 90\n1, GPU-1, 20000, 1500, 80",
        "processes": "123, GPU-0, 800\n456, GPU-0, 900",
        "stage_progress": {"stage": "training", "completed": 10, "total": 20},
    }]

    result = summarize_run_cost(manifest, telemetry)

    assert result["sampled_peak_tracked_gpu_memory_mib"] == {0: 800}
    assert result["sampled_peak_host_process_tree_rss_mib"] == 300
    assert result["concurrent_process_counts_at_start"] == {"0": 2}


def test_absent_host_memory_samples_are_unknown_not_zero():
    manifest = {
        "pid": 123,
        "seed": 72,
        "stage": "selection",
        "status": "failed",
        "physical_gpus": [],
        "logical_device": "cpu",
        "cuda_visible_devices": "",
        "initial_gpu_snapshot": None,
        "shell_command": "python run.py",
        "output": "/tmp/run",
        "source_snapshot": {"sha256": "source-hash"},
    }

    result = summarize_run_cost(manifest, [{"elapsed_seconds": 1.0}])

    assert result["sampled_peak_host_process_tree_rss_mib"] is None
    assert result["sum_of_process_tree_vm_hwm_mib"] is None

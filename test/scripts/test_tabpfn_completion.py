import pytest
from pathlib import Path
from scripts.research.complete_tabpfn import verify_command


def test_reuse_allows_only_output_name_change():
    verify_command(['python', 'picid/run.py', 'seed=72', 'model.fit_mode=low_memory', 'experiment_group=old'],
                   ['python', 'picid/run.py', 'seed=72', 'model.fit_mode=low_memory', 'experiment_group=new'])


@pytest.mark.parametrize('change', ['model.fit_mode=cached', 'seed=88', 'test=true', 'model.max_fit_samples=10000'])
def test_reuse_rejects_protocol_change(change):
    with pytest.raises(ValueError, match='registered protocol'):
        verify_command(['python', 'picid/run.py', 'seed=72', 'model.fit_mode=low_memory'],
                       ['python', 'picid/run.py', 'seed=72', 'model.fit_mode=low_memory', change])


def test_memory_saving_execution_is_a_separate_recorded_override():
    from scripts.research.complete_tabpfn import overrides

    command = ["python", "picid/run.py", "seed=72", "test=false",
               "model.memory_saving_mode=true"]
    assert overrides(command)["model.memory_saving_mode"] == "true"


def test_tracker_records_environment_overrides(tmp_path):
    import json
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "tracked"
    command = [sys.executable, "scripts/research/run_tracked.py", "--output", str(output),
               "--gpu", "0", "--expected-peak-mib", "1", "--reserve-mib", "0",
               "--seed", "72", "--stage", "environment_test", "--env",
               "PYTORCH_ALLOC_CONF=expandable_segments:True", "--", sys.executable,
               "-c", "import os; assert os.environ['PYTORCH_ALLOC_CONF'] == 'expandable_segments:True'; assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'"]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["environment_overrides"] == {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in manifest["shell_command"]

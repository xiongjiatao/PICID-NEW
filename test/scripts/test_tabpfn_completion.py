from pathlib import Path

import pytest
from scripts.research.complete_tabpfn import verify_command
from scripts.research.run_tracked import directory_digest


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
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "tracked"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_smi = fake_bin / "nvidia-smi"
    fake_smi.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *--query-gpu=*) printf '0, GPU-TEST, 1024, 0, 0\\n' ;;\n"
        "  *--query-compute-apps=*) : ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    fake_smi.chmod(0o755)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    package = overlay / "tabpfn"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "mutable.pyc").write_bytes(b"ignored bytecode")
    overlay_manifest = overlay / "overlay_manifest.json"
    overlay_manifest.write_text(json.dumps({"patched_package_sha256": directory_digest(package)}) + "\n")
    command = [sys.executable, "scripts/research/run_tracked.py", "--output", str(output),
               "--gpu", "0", "--expected-peak-mib", "1", "--reserve-mib", "0",
               "--seed", "72", "--stage", "environment_test", "--env",
               "PYTORCH_ALLOC_CONF=expandable_segments:True", "--", sys.executable,
               "-c", "import os; assert os.environ['PYTORCH_ALLOC_CONF'] == 'expandable_segments:True'; assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'"]
    command[-4:-4] = ["--env", f"PYTHONPATH={overlay}"]
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    result = subprocess.run(command, cwd=repo, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["environment_overrides"] == {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": str(overlay),
    }
    overlay_record = manifest["dependency_overlays"][0]
    assert overlay_record["manifest_path"] == str(overlay_manifest)
    assert overlay_record["manifest"]["patched_package_sha256"] == directory_digest(package)
    assert overlay_record["package_sha256_verified"] == directory_digest(package)
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in manifest["shell_command"]
    assert "PYTORCH_ALLOC_CONF=expandable_segments:True" in manifest["shell_command"]

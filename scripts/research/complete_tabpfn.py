"""Finish explicitly separated TabPFN protocols with fail-closed result auditing."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from picid.research.chunks import atomic_json, file_digest
from picid.research.protocol import Candidate, PHYSICAL_GPUS, digest, freeze_selection
from scripts.research.run_selection_grid import parse_validation_metrics
from scripts.research.aggregate_seed_metrics import aggregate_seed_metrics
from scripts.research.run_tracked import gpu_snapshot


def overrides(command):
    return dict(value.split('=', 1) for value in command[2:] if '=' in value)


def verify_command(actual, expected):
    ignored = {'experiment_group'}
    a, b = overrides(actual), overrides(expected)
    if {k: v for k, v in a.items() if k not in ignored} != {
        k: v for k, v in b.items() if k not in ignored
    }:
        raise ValueError('Attempt does not match the registered protocol')


def run_stage(stage, root, gpu, python, status):
    output = Path(stage['output'])
    command = stage['command']
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_command(manifest['command'], command)
        if manifest.get('exit_code') != 0:
            raise RuntimeError(f'Existing attempt is incomplete/failed; preserve it: {output}')
        return output
    while True:
        snapshot = gpu_snapshot()
        free = {int(row.split(',')[0]): int(row.split(',')[2])
                for row in snapshot['gpus'].splitlines()}
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) // 1024
        if free[gpu] >= stage['peak_mib'] + 1024 and available >= stage.get('host_reserve_mib', 64000):
            break
        atomic_json(status, {'status': 'waiting_for_resource_budget', 'stage': str(output),
                             'gpu': gpu, 'free_gpu_mib': free[gpu], 'host_available_mib': available,
                             'updated_unix': time.time()})
        time.sleep(30)
    atomic_json(status, {'status': 'running', 'stage': str(output), 'gpu': gpu,
                         'updated_unix': time.time()})
    subprocess.run([python, 'scripts/research/run_tracked.py', '--output', str(output),
                    '--gpu', str(gpu), '--expected-peak-mib', str(stage['peak_mib']),
                    '--seed', overrides(command)['seed'], '--stage', stage['kind'], '--',
                    *command], cwd=root, check=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('exit_code') != 0:
        raise RuntimeError('Tracked process did not finish successfully')
    return output


def audit_final(output, root, python, dataset, seed, frozen, results):
    log = (output / 'stdout.log').read_text(errors='replace')
    locations = re.findall(r'Output dir: (.+)', log)
    if not locations:
        raise ValueError('Missing saved prediction location')
    run = Path(locations[-1].strip())
    metrics = run / 'eval_details/best_epoch/test/metrics.json'
    predictions = run / 'eval_details/test/predictions.nc'
    report = results / f'device_metrics_seed{seed}.json'
    subprocess.run([python, 'scripts/research/summarize_unit_metrics.py', '--metrics', str(metrics),
                    '--output', str(report), '--dataset', dataset, '--seed', str(seed),
                    '--config-sha256', frozen['sha256']], cwd=root, check=True)
    subprocess.run([python, 'scripts/research/audit_prediction_metrics.py', '--predictions', str(predictions),
                    '--metrics', str(report), '--output', str(results / f'prediction_audit_seed{seed}.json')],
                   cwd=root, check=True)
    return report


def execute(plan, path):
    root = Path(plan['root']); python = plan['python']; gpu = plan['gpu']
    results = Path(plan['results']); results.mkdir(parents=True, exist_ok=True)
    if gpu not in PHYSICAL_GPUS:
        raise ValueError('GPU is outside the current allow-list')
    worker = results / 'workers' / plan['worker']
    worker.mkdir(parents=True, exist_ok=True)
    status = worker / 'completion_status.json'
    identity = worker / 'completion_plan.json'
    if identity.exists() and json.loads(identity.read_text()) != plan:
        raise ValueError('Completion plan changed; use a new protocol directory')
    atomic_json(identity, plan)
    try:
        if plan['dataset'] == 'nc_p':
            records = {}; grid = []
            for stage in plan['selection']:
                candidate = Candidate('nc_p', 'tabpfn', stage['window'], stage['stride'])
                grid.append(candidate)
                try:
                    expected = [python, 'picid/run.py', *candidate.overrides(seed=72, test=False),
                                *plan['execution_overrides'], *plan['data_overrides']]
                    verify_command(stage['command'], expected)
                    output = run_stage(stage, root, gpu, python, status)
                    metrics = parse_validation_metrics(output / 'stdout.log')
                    records[candidate.key] = {'status': 'success', 'seed': 72, 'test_enabled': False,
                                              **metrics, 'manifest_sha256': file_digest(output / 'manifest.json')}
                except (subprocess.CalledProcessError, RuntimeError) as exc:
                    records[candidate.key] = {'status': 'failed', 'seed': 72, 'error': repr(exc),
                                              'output': stage['output']}
                atomic_json(results / 'selection_progress.json', records)
            frozen = freeze_selection(grid, records, plan['execution_overrides'])
            atomic_json(results / 'frozen_seed72.json', frozen)
            candidate = Candidate(**frozen['candidate'])
            stages = []
            for seed in (72, 88, 101):
                name = f"nc_p_tabpfn_balanced10k_cached_yield32_seed{seed}_final"
                command = [python, 'picid/run.py', *candidate.overrides(seed=seed, test=True),
                           f'experiment_group={name}', *plan['execution_overrides'], *plan['data_overrides']]
                stages.append({'output': str(root / 'artifacts/formal' / name), 'command': command,
                               'peak_mib': 23000, 'kind': 'frozen_final_balanced10k_cached_yield32'})
        else:
            frozen = json.loads(Path(plan['frozen']).read_text())
            if frozen['sha256'] != digest({k: v for k, v in frozen.items() if k != 'sha256'}):
                raise ValueError('Frozen configuration digest mismatch')
            if frozen['sha256'] != plan['frozen_sha256']:
                raise ValueError('Wrong frozen execution protocol')
            stages = plan['finals']
        for stage in stages:
            seed = int(overrides(stage['command'])['seed'])
            expected = [python, 'picid/run.py',
                        *Candidate(**frozen['candidate']).overrides(seed=seed, test=True),
                        *frozen['execution_overrides'], *plan['data_overrides']]
            verify_command(stage['command'], expected)
            output = run_stage(stage, root, gpu, python, status)
            seed = int(overrides(stage['command'])['seed'])
            audit_final(output, root, python, plan['dataset'], seed, frozen, results)
        # Each worker writes its own status; aggregation is safe only with all three audited seeds.
        reports = [results / f'device_metrics_seed{s}.json' for s in (72, 88, 101)]
        if all(p.exists() and (results / f'prediction_audit_seed{s}.json').exists()
               for p, s in zip(reports, (72, 88, 101))):
            atomic_json(results / 'three_seed_summary.json', aggregate_seed_metrics([json.loads(p.read_text()) for p in reports]))
        atomic_json(status, {'status': 'assigned_stages_audited', 'plan': str(path),
                             'protocol': plan['protocol'], 'updated_unix': time.time()})
    except Exception as exc:
        atomic_json(status, {'status': 'incomplete', 'error': repr(exc), 'updated_unix': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    with args.plan.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(plan, args.plan)


if __name__ == '__main__':
    main()

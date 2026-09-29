import pytest
from scripts.research.complete_tabpfn import verify_command


def test_reuse_allows_only_output_name_change():
    verify_command(['python', 'picid/run.py', 'seed=72', 'model.fit_mode=low_memory', 'experiment_group=old'],
                   ['python', 'picid/run.py', 'seed=72', 'model.fit_mode=low_memory', 'experiment_group=new'])


@pytest.mark.parametrize('change', ['model.fit_mode=cached', 'seed=88', 'test=true', 'model.max_fit_samples=10000'])
def test_reuse_rejects_protocol_change(change):
    with pytest.raises(ValueError, match='registered protocol'):
        verify_command(['python', 'picid/run.py', 'seed=72', 'model.fit_mode=low_memory'],
                       ['python', 'picid/run.py', 'seed=72', 'model.fit_mode=low_memory', change])

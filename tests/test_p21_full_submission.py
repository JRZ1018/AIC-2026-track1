"""Pure CPU tests; no test images, training artifacts or CUDA accessed."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('p21fullsubmission', ROOT / 'scripts/run_stage2_p21_full_submission.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metrics(epochs):
    return {'training_scope': 'full_train_fixed_epochs', 'train_count': 148695, 'validation_count': 0,
        'selected_checkpoint': 'final.pt', 'history': [{'epoch': e, 'train_loss': 1.2,
         'confidence_weighting': {'next_effective_quality': {'decision': 'L1'}}} for e in range(1, epochs + 1)]}


def test_complete_h1_and_full20():
    module.check_metrics(metrics(100), 100, False)
    module.check_metrics(metrics(20), 20, True)


@pytest.mark.parametrize('key,value', [('training_scope', 'fixed_split_validation'), ('train_count', 118958),
    ('validation_count', 29737), ('selected_checkpoint', 'best.pt')])
def test_rejects_fold_or_selected_best(key, value):
    m = metrics(20); m[key] = value
    with pytest.raises(ValueError): module.check_metrics(m, 20, True)


@pytest.mark.parametrize('loss', [float('nan'), float('inf'), -float('inf'), '1.2', True])
def test_rejects_invalid_losses(loss):
    m = metrics(20); m['history'][5]['train_loss'] = loss
    with pytest.raises(ValueError): module.check_metrics(m, 20, True)


def test_rejects_incomplete_and_ac_l2():
    m = metrics(19)
    with pytest.raises(ValueError): module.check_metrics(m, 20, True)
    m = metrics(20); m['history'][1]['confidence_weighting']['next_effective_quality']['decision'] = 'L2'
    with pytest.raises(ValueError): module.check_metrics(m, 20, True)


def test_frozen_command_uses_full_final_and_p7_protocol():
    command = module.command_for('a' * 64)
    assert command[command.index('--checkpoint') + 1].endswith('/final.pt') or command[command.index('--checkpoint') + 1].endswith('\\final.pt')
    assert 'full20' in command[command.index('--checkpoint') + 1]
    assert '--four-view-scale384' in command
    assert command[command.index('--balanced-prior-temperature-multiplier') + 1] == '0.5'
    assert command[command.index('--balanced-prior-bias-strength') + 1] == '0.75'
    assert command[command.index('--expected-checkpoint-sha256') + 1] == 'a' * 64
    assert len(module.full.GATES) == 2
    assert module.HARD_SECONDS == 3600 and module.MAX_ADDED == 1024**3

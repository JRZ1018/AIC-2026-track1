"""Explicit one-off exploration is not a statistical promotion or full training."""
from __future__ import annotations

MX1_LAST = 'a459c971da27073723a629bcefe23c64a2115c6a7491fecb4d690a2e5e4fe4b7'
MX1_SOURCE = 'f56f48a6688e550841f256e8602512482482b71260fa8e6fc44cae61fae9fb17'
MX1_FORMAL = '73e1a2ecbe67584b93f995e12b8009b0b420429eada97819d7bfeaa0221bd413'
H1 = 'b6e35e54fac98a11ecb2948db2de278a32aca4526cbf5f210bc3a8c2b34ac10f'


def validate_mx1_exploration(value: dict) -> None:
    expected = {
        'schema': 'aictrack1.mx1.exploration.v1',
        'explicit_user_authorization': True,
        'authorization_text': 'MX1 MixUp监督未证实稳定收益，是否可以先出一个探索包，我来测分，看到底是提高还是降低？',
        'last_sha256': MX1_LAST, 'training_source_sha256': MX1_SOURCE,
        'formal_report_sha256': MX1_FORMAL, 'head_sha256': H1,
        'checkpoint_selection': 'fixed last epoch20; never best',
        'train_count': 118958, 'validation_count': 29737,
        'full_train': False, 'screen_gate_passed': False,
        'multi_training_seed_confirmed': False,
        'maximum_test_runs': 1, 'online_submission_by_agent': False,
        'p7': {'temperature': .5, 'strength': .75, 'iterations': 60},
        'four_view': '336+flip+384crop336+flip',
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError('exploration contract mismatch: ' + key)


def validate_terminal_history(state: dict) -> None:
    if state.get('epoch') != 20:
        raise ValueError('fixed final epoch20 required')
    # The frozen MX1 trainer does not store this optional field. Its exact H1
    # is independently bound by the experiment config, source and freeze.
    if 'head_initializer_sha256' in state and state['head_initializer_sha256'] != H1:
        raise ValueError('new-fold H1 mismatch')
    history = state.get('history', [])
    if [row.get('epoch') for row in history] != list(range(1, 21)):
        raise ValueError('complete twenty-epoch history required')
    if len(state.get('trainable_state', {})) != 96:
        raise ValueError('exact original last-eight-block parameter coverage required')
    if state.get('training_scope') != 'fixed_split_validation':
        raise ValueError('fold model must never be relabeled as full training')


def inference_only_payload(state: dict) -> dict:
    """Keep all learned tensors and provenance; remove training-only state."""
    validate_terminal_history(state)
    excluded = {'optimizer', 'scheduler', 'scaler', 'python_rng_state',
                'numpy_rng_state', 'torch_rng_state', 'cuda_rng_state_all',
                'confidence_weighting', 'history', 'best_state', 'best_model'}
    payload = {key: value for key, value in state.items() if key not in excluded}
    payload['mode'] = 'last_n_blocks'
    payload['checkpoint_selection'] = 'fixed last epoch20; explicit exploratory exception; screen failed'
    return payload

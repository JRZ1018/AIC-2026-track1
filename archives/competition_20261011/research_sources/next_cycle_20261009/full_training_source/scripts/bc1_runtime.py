"""Exact audited bootstrap binding, no formal AC mutation or implicit fitting."""
from pathlib import Path
import torch
from aictrack1.methods.bootstrap_curriculum import FirstEpochCurriculum

WEIGHTS_SHA='091487385b52055c2a02e4eac9e361aed52eb5eff1095bc15e6c5c78076f07f3'
HEAD_SHA='b6e35e54fac98a11ecb2948db2de278a32aca4526cbf5f210bc3a8c2b34ac10f'
def load_curriculum(root,setting,confidence,head_sha):
    from mainline_guard import sha
    if setting is None:return None
    if setting != dict(protocol='initial-fixed336-train-ac-rank-epoch1-v1',
                       weights_sha256=WEIGHTS_SHA):
        raise ValueError('unregistered bootstrap configuration')
    if confidence is None or confidence.epochs_updated or bool(confidence.seen.any()):
        raise ValueError('pristine formal AC required')
    p=Path(root)/'inputs/bootstrap_weights.pt'
    if sha(p)!=WEIGHTS_SHA or head_sha!=HEAD_SHA:raise ValueError('bootstrap input SHA drift')
    state=torch.load(p,map_location='cpu',weights_only=True)
    if (state['head_sha256']!=head_sha or state['test_access'] is not False
            or state['row_binding']['counts']!={'train':118958,'validation':29737}):
        raise ValueError('bootstrap scope/head mismatch')
    curriculum=FirstEpochCurriculum(confidence.sample_indices,confidence.class_ids,
                                   torch.ones_like(confidence.weights),min_weight=confidence.min_weight)
    curriculum.load_state_dict(state)
    return curriculum

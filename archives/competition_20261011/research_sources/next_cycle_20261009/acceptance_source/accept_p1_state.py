"""Strict joint-head restoration checks, independently usable by P1 acceptance.

No training or model downloads are performed here. The toy tests use small
synthetic tensors and the actual frozen P1 loader helpers.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
P1 = OUT.parent/'training_source'
sys.path.insert(0,str(P1/'src'))

import torch
from aictrack1.engine.peft import configure_classifier_trainability_from_state, load_trainable_state_dict
from aictrack1.models.peft import PEFTImageClassifier


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def restore_joint_state(visual, initial_head, checkpoint, *, expected_visual_tensors=152):
    """Restore visual AND classifier before freezing parameters for inference."""
    require(checkpoint.get('mode') == 'full_visual', 'P1 joint checkpoint must be full_visual')
    require(checkpoint.get('epoch') == 20, 'P1 acceptance requires fixed epoch20, not a partial checkpoint')
    state=checkpoint.get('trainable_state')
    require(isinstance(state,dict), 'trainable_state is missing')
    head_names={f'classifier.{name}' for name,_ in initial_head.named_parameters()}
    require(head_names == {'classifier.linear.weight','classifier.linear.bias'}, 'P1 requires its original linear task head')
    require({name for name in state if name.startswith('classifier.')} == head_names, 'P1 classifier state is absent or incomplete')
    require(all(isinstance(value,torch.Tensor) and bool(torch.isfinite(value).all()) for value in state.values()), 'checkpoint contains non-finite or invalid state')
    initial={f'classifier.{name}':parameter.detach().clone() for name,parameter in initial_head.named_parameters()}
    for parameter in initial_head.parameters(): parameter.requires_grad_(False)
    enabled=configure_classifier_trainability_from_state(initial_head,state)
    require(enabled, 'joint classifier was not enabled before strict restoration')
    model=PEFTImageClassifier(visual,initial_head,normalize_features=True)
    names={name for name,parameter in model.named_parameters() if parameter.requires_grad}
    require(len(names-head_names) == expected_visual_tensors and len(names) == expected_visual_tensors+2, 'unexpected visual/head tensor count')
    load_trainable_state_dict(model,state)
    for name,parameter in model.named_parameters():
        if name in names:
            require(torch.equal(parameter,state[name]), 'loaded parameters differ from new final/last state')
    evidence=dict(visual_tensors=expected_visual_tensors,head_tensors=2,trainable_tensors=len(names),
                  head_source='new checkpoint trainable_state, not H1 initializer',
                  head_matches_checkpoint=True,
                  head_initializer_max_abs_delta={name:float((state[name]-initial[name]).abs().max()) for name in sorted(head_names)})
    return model,evidence


def restore_joint_optimizer(model,saved_optimizer, *, visual_lr=1.75e-6,head_lr=5e-6):
    """Restore the two original AdamW groups without changing their ordering."""
    visual=[parameter for name,parameter in model.named_parameters() if parameter.requires_grad and name.startswith('visual.')]
    head=[parameter for name,parameter in model.named_parameters() if parameter.requires_grad and name.startswith('classifier.')]
    require(len(head)==2 and visual, 'joint optimizer model groups are incomplete')
    groups=saved_optimizer.get('param_groups',[])
    require(len(groups)==2 and [len(group['params']) for group in groups] == [len(visual),len(head)], 'P1 optimizer must retain visual/head group order')
    optimizer=torch.optim.AdamW([dict(params=visual,lr=visual_lr),dict(params=head,lr=head_lr)],weight_decay=.01)
    optimizer.load_state_dict(saved_optimizer)
    require(len(optimizer.state)==len(visual)+len(head), 'incomplete optimizer moments')
    for params,group in zip([visual,head],groups,strict=True):
        for parameter,index in zip(params,group['params'],strict=True):
            actual=optimizer.state[parameter];expected=saved_optimizer['state'][index]
            require(set(actual)==set(expected),'optimizer moment keys changed')
            for key,value in actual.items():
                if isinstance(value,torch.Tensor): require(torch.equal(value,expected[key]) and bool(torch.isfinite(value).all()),'optimizer moment differs/nonfinite')
                else: require(value==expected[key],'optimizer scalar differs')
            require(actual['exp_avg'].shape==actual['exp_avg_sq'].shape==parameter.shape,'optimizer moment shape differs')
    return optimizer


def export_from_fixed_last(last):
    """A lightweight export must keep all 154 learned tensors, including head."""
    require(last.get('epoch')==20 and last.get('mode')=='full_visual','only fixed epoch20 full_visual can export')
    state=last.get('trainable_state',{})
    require({'classifier.linear.weight','classifier.linear.bias'}.issubset(state),'export would drop the learned task head')
    require(len(state)==154,'real P1 final export needs 152 visual and 2 head tensors')
    require(all(isinstance(value,torch.Tensor) and bool(torch.isfinite(value).all()) for value in state.values()),'invalid final state')
    return {key:value.detach().cpu().clone() for key,value in state.items()}

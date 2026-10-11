"""Positive fresh matched C/J evidence, never an exploratory-fold launch permission."""
from __future__ import annotations
import json, math
from pathlib import Path
from p1_contract import ARMS, CLIP_SHA, _asset, _digest, _paired_identity_sha, sha

FRESH_SPLIT_SHA='57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7'
RUNTIME={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
REQUIRED_EVIDENCE=('checkpoint_C','config_C','identity_C','checkpoint_audit_C',
    'checkpoint_J','config_J','identity_J','checkpoint_audit_J','paired_report',
    'training_source','diagnostic_source','selection_protocol','frozen_postprocess',
    'split','h1','h1_receipt','runtime_C','runtime_J','root_review')

def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def validate_confirmation(value):
    arm=value.get('selected_arm')
    if (value.get('schema')!='aictrack1.p1.fresh_confirmation.v1'
            or value.get('status')!='positive_confirmed_for_full_training' or arm not in ('J5','J20')
            or value.get('split_seed')!=20261009 or value.get('split_sha256')!=FRESH_SPLIT_SHA
            or value.get('train_count')!=118958 or value.get('validation_count')!=29737
            or value.get('train_query_overlap')!=0 or value.get('old_query_overlap')!=0
            or value.get('used_test_or_online_scores') is not False or value.get('test_access') is not False):
        raise ValueError('independent fresh split, legal matched C/J and positive confirmation required')
    initial=value.get('initialization',{})
    if (initial.get('clip_sha256')!=CLIP_SHA or initial.get('h1_epoch')!=100
            or initial.get('h1_scope')!='fold_train_fixed_final_epochs'):
        raise ValueError('fresh official CLIP and new train-only H1100 required')
    runtime=value.get('runtime',{})
    if (any(runtime.get(k)!=v for k,v in RUNTIME.items())
            or runtime.get('actual_torch_threads')!=4 or runtime.get('equal_between_arms') is not True):
        raise ValueError('matched measured thread4 runtime required')
    gate=value.get('decision_gate',{});review=value.get('root_review',{})
    if (gate.get('status')!='pass' or not _number(gate.get('primary_delta_pp')) or gate['primary_delta_pp']<=0
            or not _number(gate.get('natural_p7_delta_pp'))
            or review.get('status')!='pass' or review.get('reviewer')!='root'
            or not isinstance(review.get('reason'),str) or not review['reason'].strip()):
        raise ValueError('positive paired gate and explicit root review required')
    evidence=value.get('evidence',{})
    for name in REQUIRED_EVIDENCE:_asset(evidence.get(name))
    bindings={'split_sha256':'split','h1_sha256':'h1','h1_receipt_sha256':'h1_receipt',
        'confirmation_training_source_manifest_sha256':'training_source',
        'diagnostic_source_manifest_sha256':'diagnostic_source','selection_protocol_sha256':'selection_protocol',
        'frozen_postprocess_sha256':'frozen_postprocess'}
    for field,item in bindings.items():
        if not _digest(value.get(field)) or value[field]!=evidence[item]['sha256']:
            raise ValueError('fresh confirmation evidence mismatch: '+field)
    if set(value.get('arms',{}))!={'C',arm}:raise ValueError('exact matched C/J pair required')
    for name,suffix in [('C','C'),(arm,'J')]:
        data=value['arms'][name]
        if data.get('epoch')!=20 or data.get('checkpoint_role')!='last':raise ValueError('both arms must be fixed-last20')
        for field,item in [('checkpoint_sha256','checkpoint'),('config_sha256','config'),
                ('identity_sha256','identity'),('checkpoint_audit_sha256','checkpoint_audit')]:
            if data.get(field)!=evidence[item+'_'+suffix]['sha256']:raise ValueError('fresh pair identity mismatch: '+name+'.'+field)
    return value

def verify_confirmation(value):
    """Read immutable evidence and both real states before any full CUDA launch."""
    import yaml
    from p1_contract import _verify_fold_evidence
    validate_confirmation(value);e=value['evidence'];arm=value['selected_arm']
    for name,item in e.items():
        _asset(item)
        if sha(item['path'])!=item['sha256']:raise ValueError('fresh evidence drift: '+name)
    read=lambda name:json.loads(Path(e[name]['path']).read_text(encoding='utf-8'))
    if read('root_review')!=value['root_review']:raise ValueError('root review is not the immutable reviewed receipt')
    h1=read('h1_receipt')
    if (h1.get('checkpoint_sha256')!=value['h1_sha256'] or h1.get('epoch')!=100 or h1.get('complete') is not True
            or h1.get('scope')!='fold_train_fixed_final_epochs'
            or h1.get('row_binding',{}).get('counts')!={'train':118958,'validation':29737}
            or h1.get('validation_selection') is not False or h1.get('test_access') is not False):
        raise ValueError('new train-only H1 actual receipt differs')
    paired=read('paired_report')
    if (paired.get('schema')!='aictrack1.confirmation.fixed_last_paired.v1' or paired.get('p1_arm')!=arm
            or paired.get('diagnostic_source_manifest_sha256')!=value['diagnostic_source_manifest_sha256']
            or paired.get('selection_protocol_sha256')!=value['selection_protocol_sha256']
            or _paired_identity_sha(paired.get('candidate',{}))!=e['identity_J']['sha256']
            or _paired_identity_sha(paired.get('control',{}))!=e['identity_C']['sha256']):
        raise ValueError('fresh pair report binding differs')
    eligibility=paired.get('eligibility',{})
    for name in ('primary_delta_pp','natural_p7_delta_pp'):
        if eligibility.get(name)!=value['decision_gate'][name]:raise ValueError('confirmation gate metric differs: '+name)
    configs=[];runtimes=[]
    for selected,suffix in [('C','C'),(arm,'J')]:
        raw=yaml.safe_load(Path(e['config_'+suffix]['path']).read_text(encoding='utf-8'))['experiment']
        if sha(raw['split'])!=value['split_sha256'] or sha(raw['head_checkpoint'])!=value['h1_sha256']:
            raise ValueError('fresh model did not use confirmation split/H1')
        config=dict(raw);config.pop('run_id',None);config.pop('output_dir',None)
        config['training']=dict(config['training']);config['training'].pop('head_learning_rate')
        configs.append(config)
        identity=read('identity_'+suffix)
        if identity.get('split_sha256')!=value['split_sha256']:
            raise ValueError('fresh identity split differs')
        audit=read('checkpoint_audit_'+suffix)
        expected_audit=dict(schema='aictrack1.confirmation.fixed_last_diagnostic_audit.v1',
            status='complete_fixed_last20_audit',complete=True,checkpoint_role='last',checkpoint_epoch=20,epoch=20,
            fixed_last_sha256=e['checkpoint_'+suffix]['sha256'],checkpoint_sha256=e['checkpoint_'+suffix]['sha256'],
            config_sha256=e['config_'+suffix]['sha256'],split_sha256=value['split_sha256'],h1_sha256=value['h1_sha256'],
            training_source_manifest_sha256=value['confirmation_training_source_manifest_sha256'],
            diagnostic_source_manifest_sha256=value['diagnostic_source_manifest_sha256'],model_arm=selected,
            trainable_tensors=152 if selected=='C' else 154,trainable_numel=87849216 if selected=='C' else 88233966,
            classifier_head_source='fresh_fold_h1_frozen' if selected=='C' else 'joint_fixed_last_state',
            classifier_parameters_restored_equal=True,test_access=False)
        if any(audit.get(k)!=v for k,v in expected_audit.items()):raise ValueError('actual fresh fixed-last audit differs: '+selected)
        runtime=read('runtime_'+suffix)
        if (runtime.get('schema')!='aictrack1.confirmation.runtime.v1' or runtime.get('arm')!=selected
                or runtime.get('source_sha256')!=value['confirmation_training_source_manifest_sha256']
                or runtime.get('config_sha256')!=e['config_'+suffix]['sha256']
                or runtime.get('split_sha256')!=value['split_sha256'] or runtime.get('h1_sha256')!=value['h1_sha256']
                or runtime.get('test_access') is not False
                or any(runtime.get('runtime',{}).get(k)!=v for k,v in RUNTIME.items())
                or runtime.get('runtime',{}).get('torch_num_threads')!=4):
            raise ValueError('actual fresh runtime binding differs: '+selected)
        runtimes.append(runtime['runtime'])
        choice=dict(selected_arm=selected,train_count=118958,validation_count=29737,
            diagnostic_source_manifest_sha256=value['diagnostic_source_manifest_sha256'],
            selection_protocol_sha256=value['selection_protocol_sha256'],evidence={
                'checkpoint':e['checkpoint_'+suffix],'config':e['config_'+suffix],'identity':e['identity_'+suffix],
                'paired_report':e['paired_report'],'training_source':e['training_source']})
        _verify_fold_evidence(choice)
    if configs[0]!=configs[1]:raise ValueError('fresh C/J differ by more than head LR/output identity')
    if runtimes[0]!=runtimes[1]:raise ValueError('fresh measured C/J runtimes differ')
    return value

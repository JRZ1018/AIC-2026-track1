"""P1 FULL identities. A real fixed-last selection receipt is mandatory before GPU."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS={'C':0.,'J5':5e-6,'J20':2e-5}
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST_SHA='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
FULL_H1_SHA='fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605'
FOLD_SOURCE_J='2fee29fc5e5c282a331883de2b3cd4d5865c28e38a8cead89be247ec95a6c9fe'
FOLD_SOURCE_C='0f19ecb1347578f5b3eee02f048cda939b1bf70382858987297418e504a2c097'

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def _digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def _asset(item):
    if not isinstance(item,dict) or not isinstance(item.get('path'),str) or not item['path'].startswith('/') or not _digest(item.get('sha256')):
        raise ValueError('positive absolute remote path and SHA required')
    return item

def _paired_identity_sha(side):
    direct=side.get('identity_sha256');nested=side.get('provenance',{}).get('identity_sha256')
    if direct is not None and nested is not None and direct!=nested:raise ValueError('paired identity aliases disagree')
    return direct if direct is not None else nested

def validate_assets(payload):
    if payload.get('schema')!='aictrack1.p1.full_assets.v1' or payload.get('status')!='pinned_full_assets_runtime_sha_required':
        raise ValueError('full-only asset contract required')
    if (payload.get('competition_stage')!='stage2' or payload.get('train_count')!=148695 or payload.get('validation_count')!=0
            or payload.get('num_classes')!=750 or payload.get('h1_training_scope')!='full_train_fixed_epochs'
            or payload.get('h1_training_count')!=148695 or payload.get('h1_epoch')!=100
            or payload.get('h1_validation_used_for_selection') is not False):
        raise ValueError('positive full148695/validation0 H1100 binding required')
    for name in ('manifest','split','class_mapping','clip','h1','h1_receipt','label_conflicts'):_asset(payload['assets'][name])
    for key,expected in [('clip',CLIP_SHA),('manifest',MANIFEST_SHA),('h1',FULL_H1_SHA)]:
        if payload['assets'][key]['sha256']!=expected:raise ValueError('official/full identity mismatch: '+key)
    if payload['canonical_lock']!='/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json':raise ValueError('canonical lock required')
    if not payload['dataset_root'].startswith('/'):raise ValueError('absolute official train root required')
    return payload

def assets(root=ROOT):return validate_assets(json.loads((Path(root)/'inputs/full_assets.json').read_text(encoding='utf-8')))

def verify_files(root=ROOT):
    root=Path(root).resolve();manifest=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='aictrack1.p1.full_source.v1':raise ValueError('full source schema mismatch')
    for rel,digest in manifest['files'].items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts or not (root/p).resolve().is_relative_to(root):raise ValueError('source escape')
        if sha(root/p)!=digest:raise ValueError('full source drift: '+rel)
    return manifest

def verify_remote_assets(root=ROOT):
    payload=assets(root)
    for name,item in payload['assets'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('full asset drift: '+name)
    receipt=json.loads(Path(payload['assets']['h1_receipt']['path']).read_text())
    if (receipt.get('checkpoint_sha256')!=FULL_H1_SHA or receipt.get('scope')!='full_train_fixed_epochs'
            or receipt.get('epoch')!=100 or receipt.get('complete') is not True
            or receipt.get('row_binding',{}).get('counts')!={'train':148695,'validation':0}
            or receipt.get('validation_selection') is not False or receipt.get('test_access') is not False):
        raise ValueError('positive actual full H1 receipt required')
    return payload

def validate_config(raw,arm,payload):
    if arm not in ARMS or raw.get('training_scope')!='full_train_fixed_epochs' or raw.get('expected_train_count')!=148695:
        raise ValueError('positive full scope/count/arm required')
    expected=dict(epochs=20,batch_size=64,workers=8,prefetch_factor=2,persistent_workers=True,seed=payload['training_seed'],
        visual_learning_rate=1.75e-6,head_learning_rate=ARMS[arm],weight_decay=.01,gradient_clip_norm=1.)
    if raw['training']!=expected:raise ValueError('full20/LR/data-loader policy drift')
    if (raw.get('peft')!={'mode':'full_visual','lora_blocks':8}
            or raw.get('base_loss')!={'kind':'sce','alpha':.1,'beta':1.,'log_zero':-4.}
            or raw.get('selection_rule')!='fixed_final' or raw.get('confidence_observation_view')!='strong'):
        raise ValueError('full algorithm drift')
    for key,name in [('manifest','manifest'),('split','split'),('head_checkpoint','h1'),('label_conflicts','label_conflicts')]:
        if raw.get(key)!=payload['assets'][name]['path']:raise ValueError('full config/input mismatch: '+key)

def validate_selection(value,require_confirmation=True):
    expected_status='selected_for_full_training' if require_confirmation else 'selected_recipe_for_confirmation'
    if value.get('schema')!='aictrack1.p1.full_selection.v1' or value.get('status')!=expected_status:
        raise ValueError('root positive full selection or explicit confirmation recipe receipt required')
    arm=value.get('selected_arm')
    if (arm not in ('J5','J20') or value.get('head_learning_rate')!=ARMS[arm] or value.get('checkpoint_role')!='last'
            or value.get('checkpoint_epoch')!=20 or value.get('training_scope')!='fixed_split_validation'
            or not isinstance(value.get('train_count'),int) or not isinstance(value.get('validation_count'),int)
            or value['train_count']<=0 or value['validation_count']<=0 or value['train_count']+value['validation_count']!=148695
            or value.get('training_complete') is not True or value.get('decision_basis')!='current_stage_train_derived_validation'
            or value.get('used_test_or_online_scores') is not False):
        raise ValueError('complete legal heldout fixed-last20 train-only decision required')
    if not _digest(value.get('training_source_manifest_sha256')):raise ValueError('accepted selected fold source SHA required')
    for name in ('checkpoint','config','identity','paired_report','training_source'):_asset(value['evidence'][name])
    if value['evidence']['training_source']['sha256']!=value['training_source_manifest_sha256']:raise ValueError('selected source evidence differs')
    if not _digest(value.get('diagnostic_source_manifest_sha256')) or not _digest(value.get('selection_protocol_sha256')):
        raise ValueError('accepted diagnostic source/protocol SHA required')
    if not isinstance(value.get('selection_reason'),str) or not value['selection_reason'].strip():raise ValueError('selection reasoning required')
    if not isinstance(value.get('original_initial_free_bytes'),int) or value['original_initial_free_bytes']<=0:raise ValueError('original cumulative disk baseline required')
    if require_confirmation:
        item=_asset(value.get('confirmation_receipt'))
        if value.get('confirmation_receipt_sha256')!=item['sha256']:
            raise ValueError('fresh confirmation receipt positive SHA binding required')
    return value

def selection(path,expected_sha):
    if not _digest(expected_sha) or sha(path)!=expected_sha:raise ValueError('full selection input SHA mismatch')
    return validate_selection(json.loads(Path(path).read_text(encoding='utf-8')))

def _verify_fold_evidence(value):
    import torch,yaml
    evidence=value['evidence']
    for name,item in evidence.items():
        if sha(item['path'])!=item['sha256']:raise ValueError('selection evidence drift: '+name)
    identity=json.loads(Path(evidence['identity']['path']).read_text())
    if (identity.get('checkpoint_epoch')!=20 or identity.get('checkpoint_sha256')!=evidence['checkpoint']['sha256']
            or identity.get('training_scope')!='fixed_split_validation' or identity.get('holdout_verified') is not True
            or identity.get('training_count',identity.get('train_count'))!=value['train_count']
            or identity.get('validation_count')!=value['validation_count'] or identity.get('test_access') is not False):
        raise ValueError('accepted fixed-last identity mismatch')
    paired=json.loads(Path(evidence['paired_report']['path']).read_text())
    if (paired.get('schema') not in ('aictrack1.p1.fixed_last_paired.v1','aictrack1.confirmation.fixed_last_paired.v1')
            or paired.get('diagnostic_source_manifest_sha256')!=value['diagnostic_source_manifest_sha256']
            or paired.get('selection_protocol_sha256')!=value['selection_protocol_sha256']):
        raise ValueError('fixed-last paired report binding mismatch')
    side='control' if value['selected_arm']=='C' else 'candidate'
    if _paired_identity_sha(paired.get(side,{}))!=evidence['identity']['sha256']:
        raise ValueError('selected identity is not the paired-report side')
    if side=='candidate' and paired.get('p1_arm')!=value['selected_arm']:raise ValueError('paired report is for another joint arm')
    raw=yaml.safe_load(Path(evidence['config']['path']).read_text())['experiment']
    selected_source=json.loads(Path(evidence['training_source']['path']).read_text())
    if evidence['config']['sha256'] not in selected_source.get('files',{}).values():raise ValueError('selected config missing from source manifest')
    if (raw['training']['epochs']!=20 or raw['training']['head_learning_rate']!=ARMS[value['selected_arm']]
            or raw['peft']['mode']!='full_visual' or raw['training_scope']!='fixed_split_validation'):
        raise ValueError('selected source configuration mismatch')
    state=torch.load(evidence['checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
    if sha(raw['split'])!=state.get('split_sha256') or state.get('split_sha256')!=identity.get('split_sha256'):
        raise ValueError('selected heldout split identity differs')
    initial_head=torch.load(raw['head_checkpoint'],map_location='cpu',weights_only=True)
    binding=initial_head.get('row_binding',{})
    if (initial_head.get('training_scope')=='full_train_fixed_epochs'
            or binding.get('counts')!={'train':value['train_count'],'validation':value['validation_count']}
            or binding.get('validation_used_for_optimization_or_selection') is not False):
        raise ValueError('selected heldout model was initialized with a leaking/non-bound head')
    tensors=state['trainable_state'];joint=value['selected_arm']!='C'
    expected_head={'classifier.linear.weight','classifier.linear.bias'} if joint else set()
    if (state.get('epoch')!=20 or state.get('config_sha256')!=evidence['config']['sha256']
            or state.get('training_scope')!='fixed_split_validation' or state.get('mode')!='full_visual'
            or len(tensors)!=(154 if joint else 152)
            or sum(t.numel() for t in tensors.values())!=(88233966 if joint else 87849216)
            or {name for name in tensors if name.startswith('classifier.')}!=expected_head
            or state['scheduler']['T_max']!=20 or state['scheduler']['last_epoch']!=20
            or state['confidence_weighting']['epochs_updated']!=20
            or state['confidence_weighting']['sample_indices'].numel()!=value['train_count']
            or [r['epoch'] for r in state['history']]!=list(range(1,21))):
        raise ValueError('selected model is not a complete fixed20 legal fold state')
    for name,expected in [('train_manifest_sha256',MANIFEST_SHA),('clip_weight_sha256',CLIP_SHA),('class_mapping_sha256',MAPPING_SHA)]:
        if state.get(name)!=expected:raise ValueError('selected model official provenance differs')
    if any(not torch.isfinite(v).all() for v in tensors.values()):raise ValueError('nonfinite selected state')
    groups=state['optimizer']['param_groups'];expected_groups=[152,2] if joint else [152]
    expected_lrs=[1.75e-6,ARMS[value['selected_arm']]] if joint else [1.75e-6]
    if ([len(g['params']) for g in groups]!=expected_groups
            or len(state['optimizer']['state'])!=sum(expected_groups)
            or state['scheduler']['base_lrs']!=expected_lrs):
        raise ValueError('selected complete optimizer/head/LR coverage mismatch')
    if any(isinstance(v,torch.Tensor) and not torch.isfinite(v).all()
            for slot in state['optimizer']['state'].values() for v in slot.values()):raise ValueError('nonfinite selected optimizer')
    return value

def verify_selection_evidence(value,require_confirmation=True):
    validate_selection(value,require_confirmation=require_confirmation)
    if require_confirmation:
        from full_confirmation import verify_confirmation
        item=value['confirmation_receipt']
        if sha(item['path'])!=item['sha256']:raise ValueError('fresh confirmation receipt drift')
        confirmation=verify_confirmation(json.loads(Path(item['path']).read_text(encoding='utf-8')))
        e=confirmation['evidence']
        if (confirmation['selected_arm']!=value['selected_arm']
                or confirmation['confirmation_training_source_manifest_sha256']!=value['training_source_manifest_sha256']
                or confirmation['diagnostic_source_manifest_sha256']!=value['diagnostic_source_manifest_sha256']
                or confirmation['selection_protocol_sha256']!=value['selection_protocol_sha256']):
            raise ValueError('full choice is not the positively confirmed fresh recipe')
        for name,suffix in [('checkpoint','checkpoint_J'),('config','config_J'),('identity','identity_J'),
                ('paired_report','paired_report'),('training_source','training_source')]:
            if value['evidence'][name]!=e[suffix]:raise ValueError('full selection must use actual fresh confirmation evidence: '+name)
        return value
    return _verify_fold_evidence(value)

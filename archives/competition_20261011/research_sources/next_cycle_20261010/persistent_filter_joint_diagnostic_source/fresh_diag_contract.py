"""CPU PF30 v2 restoration and applied-supervision contracts on used57189 fold."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SPLIT_SHA = '57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7'
OLD_SPLIT_SHA = '0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST_SHA = '1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA = '811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
ROWS_SHA = 'ba299a60bf7cf7150e050d551d52e1979c5577a15ffd781ae874e7137f396954'
ROW_BINDING = dict(schema='aictrack1.mainline.h1.rows.v1', counts=dict(train=118958, validation=29737),
 all_rows_sha256='37b23205175174328fafbcfe18e9893df7eacc36a0580d56064281ce3eda7065',
 train_rows_sha256='1196a5653fcacd1f1387f0be84b540dedc2ae4f85b585d118655b73cb206f524',
 validation_rows_sha256='42c7e7fe4931152790e726d866eb758542f21b61c8115769f8f7f78daa57d767',
 validation_used_for_optimization_or_selection=False)
VIEW_PROTOCOL = ('native336 center and exact flip plus resize_short_side384_center_crop336 '
 'and exact flip; one checkpoint; four float logits equally averaged')
HEAD_KEYS = {'classifier.linear.weight', 'classifier.linear.bias'}
VISUAL_KEYS, JOINT_KEYS, VISUAL_NUMEL, JOINT_NUMEL = 152,154,87849216,88233966
ARMS = {'PF30':5e-6,'J5':5e-6}
H1_SHA='e88a1cbbaa8d9ebdbf1c7cf2707ac434cf440fac8bd5b4a5dc57893ba0fe15be'
H1_RECEIPT_SHA='06c762c1828915fd579fcc338ade3177f5eff96be742c9650e265845ce6f7d73'
H1_SOURCE_SHA='448a2b2a293cb3a7a51f7bd9b5fad7afde9f2cae196bc951485dedfa98511d2c'
ROUTES={'S':'S','J5_GRAPH':'J5'}
PF_SCIENCE_FILES={'configs/experiment/p1_PF30.yaml': 'c7201bdd8249f98c29b227ebde28799838e7fdf4a2a85da6103e3a6f1aa4c172', 'scripts/train_n1_peft.py': '6b8a780e990a56774b84e73dd3e8618779720867a68e9797ccdae76b821cdca6', 'scripts/p1_contract.py': '4cc6be1ee79375bfa4216832e37fe4c1f1fc2596b35cae55a9d53881c5633d3c', 'src/aictrack1/methods/persistent_filter.py': '18d9ad67a6cf70e2ae6b52fecf1a179ec46426380742ab6bd92509ccc8dcfa87', 'inputs/pf2_protocol.json': '96ca446cafe75f89136ed617c6db284a190fe7eb21657796bb8cc4121ffbb2c3', 'scripts/train_peft.py': '0758b989a1bb33ba851e03f9a852cf35b7ad30847ccbfb62cf528f8dc015a765'}
BINDING=json.loads((ROOT/'inputs/diagnostic_binding.json').read_text(encoding='utf-8'))
TRAINING_SOURCE_SHA=BINDING['training_source_sha256']
TRAIN_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_persistent_filter_joint_20261010')
REMOTE_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_persistent_filter_joint_diagnostic_20261010')


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def save_json(path,value):
 path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
 temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)

def valid_digest(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)

def confirmation_rows(path=ROOT/'inputs/confirmation_rows.csv'):
 if sha(path)!=ROWS_SHA:raise ValueError('fresh connected rows SHA mismatch')
 with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=148695 or len({r['sample_id'] for r in rows})!=148695:raise ValueError('fresh complete rows')
 parts={k:[r for r in rows if r['split']==k] for k in ('train','validation')}
 if [len(parts[k]) for k in parts]!=[118958,29737]:raise ValueError('fresh partition counts')
 if {r['group'] for r in parts['train']}&{r['group'] for r in parts['validation']}:raise ValueError('connected group overlap')
 return rows,parts

def validate_state(state,*,arm,visual_names=None,num_classes=750,feature_dim=512,visual_count=VISUAL_KEYS,visual_numel=VISUAL_NUMEL):
 if not isinstance(state,dict) or not state:raise ValueError('missing trainable state')
 names=set(state);heads={n for n in names if n.startswith('classifier.')};expected=set() if arm in ('C','S') else HEAD_KEYS
 if arm not in ARMS or heads!=expected:raise ValueError('C frozen H1 or J complete joint head key mismatch')
 visual=names-heads
 if len(visual)!=visual_count or any(not n.startswith('visual.') for n in visual):raise ValueError('visual keys/count')
 if visual_names is not None and visual!=set(visual_names):raise ValueError('not all visual tensors restored')
 if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):raise ValueError('complete finite FP32 master state required')
 if sum(state[n].numel() for n in visual)!=visual_numel:raise ValueError('visual numel')
 if heads and (tuple(state['classifier.linear.weight'].shape)!=(num_classes,feature_dim) or tuple(state['classifier.linear.bias'].shape)!=(num_classes,)):raise ValueError('head shapes')
 return dict(trainable_tensors=len(names),trainable_numel=sum(v.numel() for v in state.values()),visual_tensors=len(visual),head_tensors=len(heads))

def validate_checkpoint_metadata(state,metrics,experiment,*,arm,config_sha,last_sha,registered):
 h1=registered['assets']['h1']['sha256']
 expected=dict(competition_stage='stage2',training_scope='fixed_split_validation',mode='full_visual',visual_input_resolution=336,
 position_interpolation='bicubic_7x7_to_10x10',epoch=20,config_sha256=config_sha,train_manifest_sha256=MANIFEST_SHA,
 split_sha256=SPLIT_SHA,class_mapping_sha256=MAPPING_SHA,clip_weight_sha256=CLIP_SHA,num_classes=750,
 fresh_route_arm=arm,fresh_route_source_manifest_sha256=registered['training_source_manifest_sha256'],
 selected_route=registered['selected_route'])
 for key,value in expected.items():
  if state.get(key)!=value:raise ValueError('fresh fixed-last metadata: '+key)
 if (metrics.get('selection_rule')!='fixed final epoch 20; no validation selection' or metrics.get('selected_checkpoint')!='final.pt' or metrics.get('config_sha256')!=config_sha or metrics.get('last_checkpoint_sha256')!=last_sha or metrics.get('training_scope')!='fixed_split_validation'
     or metrics.get('train_count')!=118958 or metrics.get('validation_count')!=29737 or metrics.get('early_stop') is not None):raise ValueError('terminal metrics')
 history=state.get('history')
 if not isinstance(history,list) or history!=metrics.get('history') or [r.get('epoch') for r in history]!=list(range(1,21)) or any('validation' not in r for r in history):raise ValueError('twenty complete epochs')
 train=experiment.get('training',{})
 if (arm not in ARMS or train.get('epochs')!=20 or train.get('seed')!=20261008
     or train.get('visual_learning_rate')!=1.75e-6 or train.get('head_learning_rate')!=ARMS[arm]
     or experiment.get('selection_rule')!='fixed_final' or state.get('token_readout')!=experiment.get('token_readout')):raise ValueError('fresh recipe')
 if experiment.get('base_loss')!={'kind':'sce','alpha':.1,'beta':1.,'log_zero':-4.}:raise ValueError('typed recipe loss')
 # The real fixed-last producer records its initializer through config/terminal metrics;
 # the selected-final export's optional initializer key is not invented for last.pt.
 if ('head_initializer_sha256' in state and state['head_initializer_sha256']!=h1) or sha(experiment['head_checkpoint'])!=h1:raise ValueError('actual fresh H1 initializer')
 head_source='new_fold_H1_frozen' if arm in ('C','S') else 'joint_fixed_last_state'
 if state.get('classifier_head_source')!=head_source or (arm in ('C','S') and state.get('classifier_head_restored_equal') is not True):raise ValueError('fresh checkpoint head provenance')
 if metrics.get('head_initializer',{}).get('sha256')!=h1 or metrics.get('base_loss')!=experiment.get('base_loss') or metrics.get('trainable_total_parameters')!=(VISUAL_NUMEL if arm in ('C','S') else JOINT_NUMEL):raise ValueError('fresh terminal head/loss/parameter count')
 if arm=='PF30' and experiment.get('persistent_filter')!=dict(enabled=True,protocol='persistent-supervision-gate-v2',max_ac_weight=.30):raise ValueError('registered PF30 v2 only')
 for forbidden in ('bootstrap_curriculum_state','trainable_ema','elr_state','josrc_teacher','josrc_support'):
  if forbidden in state:raise ValueError('nonconfirmation algorithm state')
 return dict(checkpoint_role='last',checkpoint_epoch=20,epoch=20,fixed_last_sha256=last_sha,checkpoint_sha256=last_sha,
 config_sha256=config_sha,split_sha256=SPLIT_SHA,h1_sha256=h1,train_count=118958,validation_count=29737,training_seed=20261008,history_epochs=20)

def validate_optimizer_state(state,names,*,visual_names,arm):
 optimizer=state.get('optimizer',{});groups=optimizer.get('param_groups',[])
 ordered=[n for n in names if n in set(visual_names)]+[n for n in names if n in HEAD_KEYS]
 expected=[len(visual_names)] if arm in ('C','S') else [len(visual_names),2]
 if [len(g['params']) for g in groups]!=expected:raise ValueError('optimizer groups/head policy')
 lrs=[1.75e-6] if arm in ('C','S') else [1.75e-6,5e-6]
 for group,lr in zip(groups,lrs,strict=True):
  if tuple(group.get('betas',()))!=(.9,.999) or group.get('eps')!=1e-8 or group.get('weight_decay')!=.01 or group.get('initial_lr')!=lr or not isinstance(group.get('lr'),(float,int)) or abs(group['lr'])>1e-12:raise ValueError('original AdamW group recipe/terminal Cos20 LR')
 if state.get('scheduler',{}).get('base_lrs')!=lrs:raise ValueError('original visual/head Cos20 base LRs')
 ids=[i for g in groups for i in g['params']]
 if len(set(ids))!=len(ids) or len(optimizer.get('state',{}))!=len(names):raise ValueError('optimizer state IDs')
 for name,pid in zip(ordered,ids,strict=True):
  entry=optimizer['state'].get(pid,{})
  for field in ('exp_avg','exp_avg_sq'):
   v=entry.get(field)
   if not isinstance(v,torch.Tensor) or v.shape!=state['trainable_state'][name].shape or not torch.isfinite(v).all():raise ValueError('optimizer moment '+name)
  step=entry.get('step');step=float(step) if isinstance(step,torch.Tensor) else step
  if not isinstance(step,(float,int)) or not math.isfinite(step) or step<=0:raise ValueError('optimizer step')
 if state.get('scheduler',{}).get('T_max')!=20 or state['scheduler'].get('last_epoch')!=20:raise ValueError('terminal scheduler')
 scale=state.get('scaler',{}).get('scale')
 if not isinstance(scale,(float,int)) or not math.isfinite(scale) or scale<=0:raise ValueError('AMP scaler')
 if state.get('confidence_weighting',{}).get('epochs_updated')!=20:raise ValueError('AC twenty observations')
 return dict(optimizer_parameter_groups=expected,optimizer_moments_finite=True,scheduler_epoch=20,ac_observation_epochs=20)

def validate_rng_state(state):
 """Restore private CPU RNG objects only; CUDA payload is checked without CUDA access."""
 import random
 try:
  py=state['python_rng_state']; private=random.Random();private.setstate(py)
  raw=state['numpy_rng_state']
  if not isinstance(raw,(list,tuple)) or len(raw)!=5 or raw[0]!='MT19937' or not isinstance(raw[1],list) or len(raw[1])!=624 or any(not isinstance(v,int) or not 0<=v<2**32 for v in raw[1]):raise ValueError('safe-list MT19937 state required')
  if not isinstance(raw[2],int) or not 0<=raw[2]<=624 or raw[3] not in (0,1) or not isinstance(raw[4],(float,int)) or not math.isfinite(raw[4]):raise ValueError('MT19937 position/cache malformed')
  np.random.RandomState().set_state((raw[0],np.array(raw[1],dtype=np.uint32),raw[2],raw[3],raw[4]))
  cpu=state['torch_rng_state']
  if not isinstance(cpu,torch.Tensor) or cpu.device.type!='cpu' or cpu.dtype!=torch.uint8 or cpu.ndim!=1 or not cpu.numel():raise ValueError('CPU torch RNG byte state required')
  torch.Generator(device='cpu').set_state(cpu)
  cuda=state['cuda_rng_state_all']
  if not isinstance(cuda,(tuple,list)) or len(cuda)!=1 or any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.uint8 or v.ndim!=1 or not v.numel() for v in cuda):raise ValueError('one real GPU RNG byte payload required')
 except (KeyError,TypeError,RuntimeError,OverflowError) as exc:raise ValueError('RNG payload missing or malformed') from exc
 return dict(python_rng_private_restore=True,numpy_safe_list_private_restore=True,torch_rng_private_CPU_restore=True,cuda_rng_payloads=1,cuda_rng_payload_structurally_checked=True,cuda_accessed=False)

def validate_ac_state(confidence,sample_indices,class_ids,raw):
 """Restore the actual training class, retaining its exact ID/hyperparameter checks."""
 from aictrack1.methods.confidence import ClasswiseHistoryWeighting
 restored=ClasswiseHistoryWeighting(torch.as_tensor(sample_indices),torch.as_tensor(class_ids),
  **{k:raw[k] for k in ('momentum','history_mix','min_weight','magnitude_mix','tail_class_fraction',
   'tail_min_mean_weight','min_effective_mass_per_class') if k in raw})
 restored.load_state_dict(confidence)
 if any(isinstance(v,torch.Tensor) and v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in confidence.values()):raise ValueError('nonfinite full AC state')
 if restored.epochs_updated!=20 or not bool(torch.all(restored.seen==1)):raise ValueError('AC incomplete twenty-epoch coverage')
 return restored

def load_cache(bundle_path,identity_path):
 identity_path=Path(identity_path).resolve();i=json.loads(identity_path.read_text());auditpath=identity_path.parent/i['source_provenance_receipt']
 if (i.get('schema')!='aictrack1.p0_frozen_fold_input.v1' or i.get('holdout_verified') is not True
     or i.get('checkpoint_role')!='last' or i.get('checkpoint_epoch')!=20 or i.get('split_sha256')!=SPLIT_SHA
     or i.get('training_scope')!='fixed_split_validation' or i.get('test_access') is not False
     or i.get('training_validation_overlap_rows')!=0 or i.get('stage')!=2):raise ValueError('new fixed-last20 verified holdout only')
 if sha(auditpath)!=i['source_receipt_sha256'] or sha(bundle_path)!=i['bundle_sha256']:raise ValueError('cache/audit SHA')
 audit=json.loads(auditpath.read_text());arm=i['model_arm'];count,numel=(VISUAL_KEYS,VISUAL_NUMEL) if arm in ('C','S') else (JOINT_KEYS,JOINT_NUMEL)
 for field in ('checkpoint_sha256','config_sha256','split_sha256','h1_sha256','model_arm','training_source_manifest_sha256','diagnostic_source_manifest_sha256','selected_route','admission_receipt_sha256','source_h1_manifest_sha256','classifier_head_source'):
  expected=i['experiment_config_sha256'] if field=='config_sha256' else i.get(field)
  if audit.get(field)!=expected:raise ValueError('identity/audit binding: '+field)
 if (audit.get('schema')!='aictrack1.fresh_route.fixed_last_diagnostic_audit.v1' or audit.get('complete') is not True
     or audit.get('status')!='complete_fixed_last20_audit' or audit.get('checkpoint_epoch')!=20
     or audit.get('trainable_tensors')!=count or audit.get('trainable_numel')!=numel
     or audit.get('classifier_parameters_restored_equal') is not True or audit.get('model_frozen_eval') is not True
     or audit.get('exact_ac_train_id_label_binding') is not True or audit.get('inference_optimizer_steps')!=0):raise ValueError('restoration audit')
 if audit.get('checkpoint_role')!='last' or audit.get('fixed_last_sha256')!=i['checkpoint_sha256'] or audit.get('epoch')!=20 or audit.get('test_access') is not False:raise ValueError('terminal audit role/aliases')
 if audit.get('numel')!=audit['trainable_numel']:raise ValueError('numel alias conflict')
 if arm=='PF30':
  ref=i.get('persistent_filter_audit',{})
  if ref!=audit.get('persistent_filter_audit') or ref.get('protocol')!='persistent-supervision-gate-v2' or ref.get('max_ac_weight')!=.30 or ref.get('applied_epochs_checked')!=20:raise ValueError('PF audit binding')
  pfpath=identity_path.parent/ref['file']
  if Path(ref['file']).is_absolute() or '..' in Path(ref['file']).parts or sha(pfpath)!=ref['sha256']:raise ValueError('PF audit file SHA')
  pf=json.loads(pfpath.read_text())
  if pf.get('complete') is not True or pf.get('checkpoint_sha256')!=i['checkpoint_sha256'] or pf.get('training_source_sha256')!=i['training_source_manifest_sha256']:raise ValueError('PF exact checkpoint audit')
 runtime_path=identity_path.parent/i['runtime_receipt']
 if sha(runtime_path)!=i['runtime_receipt_sha256'] or i['runtime_receipt_sha256']!=audit.get('runtime_receipt_sha256'):raise ValueError('runtime receipt SHA')
 runtime=json.loads(runtime_path.read_text())
 for k,v in dict(schema='aictrack1.fresh_route.diagnostic_runtime.v1',arm=arm,source_manifest_sha256=i['training_source_manifest_sha256'],config_sha256=i['experiment_config_sha256'],split_sha256=SPLIT_SHA,h1_sha256=i['h1_sha256'],selected_route=i['selected_route'],test_access=False).items():
  if runtime.get(k)!=v:raise ValueError('actual runtime ledger '+k)
 if runtime.get('runtime')!=i.get('runtime') or runtime.get('runtime')!=audit.get('runtime'):raise ValueError('runtime aliases')
 cache=torch.load(bundle_path,map_location='cpu',weights_only=True,mmap=True)
 if (cache.get('schema')!='aictrack1.stage2_train_derived_fused_logits.v1' or 'train-derived' not in cache.get('scope','')
     or cache.get('checkpoint_role')!='final' or cache.get('checkpoint_epoch')!=20):raise ValueError('cache schema/terminal role')
 for field in ('checkpoint_sha256','split_sha256','manifest_sha256','class_mapping_sha256','experiment_config_sha256','view_protocol'):
  if cache.get(field)!=i.get(field):raise ValueError('cache identity '+field)
 scores=cache['logits'].float().numpy();targets=cache['targets'].long().numpy();counts=cache['train_class_counts'].long().numpy()
 if cache['logits'].dtype!=torch.float16 or scores.shape!=(29737,750) or targets.shape!=(29737,) or counts.shape!=(750,) or not np.isfinite(scores).all() or np.any(counts<=0) or int(counts.sum())!=118958 or np.any(targets<0) or np.any(targets>=750):raise ValueError('cache scores/counts')
 path=identity_path.parent/i['row_csv']
 if sha(path)!=i['row_csv_sha256']:raise ValueError('row CSV SHA')
 with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 _,parts=confirmation_rows();expected=parts['validation'];ids=[r['sample_id'] for r in rows];groups=[r['group'] for r in rows]
 if (len(set(ids))!=len(ids) or ids!=list(cache['sample_ids']) or ids!=[r['sample_id'] for r in expected]
     or groups!=[r['group'] for r in expected] or not np.array_equal(targets,[int(r['target']) for r in expected])
     or not np.array_equal(targets,[int(r['target']) for r in rows])):raise ValueError('fresh exact row/group/target binding')
 if hashlib.sha256('\n'.join(ids).encode()).hexdigest()!=i['sample_ids_order_sha256']:raise ValueError('ID order SHA')
 expectedcounts=np.bincount([int(r['target']) for r in parts['train']],minlength=750)
 if not np.array_equal(counts,expectedcounts):raise ValueError('fresh training prior')
 mismatches=int(np.sum(scores.argmax(1)!=[int(r['four_view_prediction']) for r in rows]))
 provenance=dict(identity_path=str(identity_path),identity_sha256=sha(identity_path),bundle_sha256=sha(bundle_path),
 checkpoint_sha256=i['checkpoint_sha256'],checkpoint_audit_sha256=sha(auditpath),cached_logits_dtype='torch.float16',decode_dtype='float32',
 fp16_argmax_mismatches_vs_original_fp32=mismatches,precision_note='FP32 conversion cannot restore discarded FP16 precision')
 return dict(scores=scores,targets=targets,train_counts=counts,ids=ids,groups=groups,identity=i,provenance=provenance)


def verify_owned_sources(root=ROOT):
 root=Path(root).resolve();path=root/'source_manifest.json';m=json.loads(path.read_text(encoding='utf-8'))
 if m.get('schema')!='aictrack1.persistent_filter_joint.exploration_diagnostic.source.v1':raise ValueError('new exploratory diagnostic source required')
 for rel,digest in m['files'].items():
  p=(root/rel).resolve()
  if Path(rel).is_absolute() or '..' in Path(rel).parts or not p.is_relative_to(root) or sha(p)!=digest:raise ValueError('diagnostic source drift/escape '+rel)
 return sha(path)

def verify_training_binding(root,expected_sha,*,runtime_assets=False,binding_only=False):
 root=Path(root).resolve();p=root/'source_manifest.json'
 if not valid_digest(expected_sha) or sha(p)!=expected_sha:raise ValueError('exact actual PF training source SHA required')
 if not binding_only and BINDING['status']=='bound':
  if expected_sha!=TRAINING_SOURCE_SHA or sha(ROOT/'inputs/training_source_manifest.json')!=expected_sha:raise ValueError('bound actual PF source snapshot')
 if runtime_assets and not binding_only and BINDING['status']!='bound':raise ValueError('actual metadata binding required before model access')
 m=json.loads(p.read_text(encoding='utf-8'));a=json.loads((root/'inputs/p1_assets.json').read_text(encoding='utf-8'))
 if m.get('schema')!='aictrack1.persistent_filter_joint.source.v1' or m.get('status')!='bound' or m.get('arms')!=['PF30'] or m.get('expected_epochs')!=20 or m.get('old_query_reused_for_exploration') is not True:raise ValueError('PF30 fixed20 exploration producer required')
 for rel,digest in m['files'].items():
  item=(root/rel).resolve()
  if Path(rel).is_absolute() or '..' in Path(rel).parts or not item.is_relative_to(root) or sha(item)!=digest:raise ValueError('training source drift/escape '+rel)
 if any(m['files'].get(k)!=v for k,v in PF_SCIENCE_FILES.items()):raise ValueError('registered PF30 science/config identity')
 if a['assets']['h1']['sha256']!=H1_SHA or a['assets']['split']['sha256']!=SPLIT_SHA or a['assets']['clip']['sha256']!=CLIP_SHA or a['assets']['manifest']['sha256']!=MANIFEST_SHA or a.get('h1_row_binding')!=ROW_BINDING:raise ValueError('legal same fold H1/CLIP/rows binding')
 if a.get('source_h1_manifest_sha256')!=H1_SOURCE_SHA or a.get('train_count')!=118958 or a.get('validation_count')!=29737 or a.get('test_access') is not False:raise ValueError('same official train-only exploration assets')
 if runtime_assets:
  for name,item in a['assets'].items():
   if sha(item['path'])!=item['sha256']:raise ValueError('actual external asset drift '+name)
 return a,m

def validate_pair(candidate,control):
 c,j=control['identity'],candidate['identity']
 if c.get('model_arm')!='J5' or j.get('model_arm')!='PF30':raise ValueError('actual old J5 and one PF30 candidate required')
 if c.get('training_source_manifest_sha256')!='4fc8e1420cf9347e79f7a0f607cd49a2f28735900de52477091324c06a0b8482' or j.get('training_source_manifest_sha256')!=TRAINING_SOURCE_SHA:raise ValueError('honest separate actual training sources required')
 for field in ('split_sha256','manifest_sha256','class_mapping_sha256','h1_sha256','view_protocol','source_h1_manifest_sha256'):
  if c.get(field)!=j.get(field) or c.get(field) is None:raise ValueError('scientific paired identity '+field)
 if c.get('runtime')!=j.get('runtime') or c.get('runtime',{}).get('torch_num_threads')!=4:raise ValueError('actual runtime mismatch')
 if any(i.get('trainable_tensors')!=154 or i.get('classifier_head_source')!='joint_fixed_last_state' for i in (c,j)):raise ValueError('complete joint heads required')
 if candidate['ids']!=control['ids'] or candidate['groups']!=control['groups'] or not np.array_equal(candidate['targets'],control['targets']) or not np.array_equal(candidate['train_counts'],control['train_counts']):raise ValueError('exact row/group/prior mismatch')
 return True


def validate_pf_supervision_state(state,indices,labels,experiment,*,expected_epochs=20,num_classes=750):
 """Strict v2 gate restore and applied epoch ledger, never equating next-mask with applied."""
 from aictrack1.methods.persistent_filter import PersistentTrainingGate
 recipe=dict(enabled=True,protocol='persistent-supervision-gate-v2',max_ac_weight=.30)
 if experiment.get('persistent_filter')!=recipe or state.get('fresh_route_arm')!='PF30':raise ValueError('PF30 registered v2 recipe')
 ix=torch.as_tensor(indices,dtype=torch.long);ys=torch.as_tensor(labels,dtype=torch.long)
 if ix.ndim!=1 or ys.shape!=ix.shape or ys.numel()==0 or int(ys.min())<0 or int(ys.max())>=num_classes:raise ValueError('complete PF train binding')
 counts=torch.bincount(ys,minlength=num_classes).tolist()
 if any(n<=0 for n in counts):raise ValueError('all train classes require positive support')
 confidence=state.get('confidence_weighting',{});weights=confidence.get('weights')
 gate=PersistentTrainingGate(ix,ys,max_ac_weight=.30)
 gate.load_state_dict(state['persistent_filter_state'],ac_weights=weights)
 if gate.epochs_updated!=expected_epochs or state.get('epoch')!=expected_epochs or confidence.get('epochs_updated')!=expected_epochs:raise ValueError('terminal gate/AC epoch')
 history=state.get('history',[])
 if not isinstance(history,list) or [r.get('epoch') for r in history]!=list(range(1,expected_epochs+1)):raise ValueError('complete applied PF history')
 def integer(v):return isinstance(v,int) and not isinstance(v,bool)
 def near(a,b):return isinstance(a,(float,int)) and not isinstance(a,bool) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-7)
 def mass(row,upper):
  original=row.get('original_AC_mass' if upper else 'original_ac_mass')
  retained=row.get('retained_AC_mass' if upper else 'retained_ac_mass')
  if not isinstance(original,(float,int)) or isinstance(original,bool) or not math.isfinite(original) or original<=0:raise ValueError('positive finite original AC mass')
  if not isinstance(retained,(float,int)) or isinstance(retained,bool) or not math.isfinite(retained) or retained<0 or retained>original+1e-7:raise ValueError('retained AC mass')
  removed=original-retained
  if removed>.10*original+1e-7:raise ValueError('class AC removal cap')
  if upper and (not near(row.get('removed_AC_mass'),removed) or not near(row.get('removed_AC_mass_fraction'),removed/original)):raise ValueError('applied mass/fraction aliases')
  return float(original),float(removed),float(retained)
 previous=None;summary=[];successful=0
 for epoch,row in enumerate(history,1):
  detail=row.get('persistent_filter',{});actual=detail.get('actual',{});nxt=detail.get('next',{})
  if any(not integer(actual.get(k)) for k in ('mask_source_completed_epoch','observed','supervised','excluded')):raise ValueError('actual global integer ledger')
  if detail.get('max_ac_weight')!=.30 or actual.get('mask_role')!='actual_epoch_supervision_mask' or actual.get('mask_source_completed_epoch')!=epoch-1 or not valid_digest(actual.get('mask_sha256')):raise ValueError('applied mask role/source/hash')
  if detail.get('observations')!=len(ix) or actual.get('observed')!=len(ix) or detail.get('supervised_samples')!=actual.get('supervised') or not integer(detail.get('skipped_optimizer_batches')) or detail['skipped_optimizer_batches']<0:raise ValueError('full epoch observations and actual optimizer supervision')
  ar=actual.get('classes',[]);nr=nxt.get('classes',[])
  if [r.get('label') for r in ar]!=list(range(num_classes)) or [r.get('label') for r in nr]!=list(range(num_classes)):raise ValueError('all ordered class ledgers')
  if any(not integer(nxt.get(k)) for k in ('epochs_updated','sample_count','excluded')):raise ValueError('NEXT integer ledger')
  if nxt.get('epochs_updated')!=epoch or nxt.get('sample_count')!=len(ix):raise ValueError('NEXT ledger completed epoch')
  original=removed=retained=0.;exclusions=supervised=next_exclusions=0
  for label,(a,n) in enumerate(zip(ar,nr,strict=True)):
   count=counts[label];cap=min(math.floor(.10*count),max(0,count-16))
   if any(not integer(a.get(k)) for k in ('observed','supervised','excluded')) or a['observed']!=count or a['supervised']+a['excluded']!=count or not 0<=a['excluded']<=cap:raise ValueError('applied class coverage/cap')
   if any(not integer(n.get(k)) for k in ('count','retained','excluded')) or n['count']!=count or n['retained']+n['excluded']!=count or not 0<=n['excluded']<=cap:raise ValueError('NEXT class coverage/cap')
   am=mass(a,True);nm=mass(n,False)
   if previous is None:
    if a['excluded']!=0 or not near(am[0],count) or not near(am[1],0):raise ValueError('first epoch original all-one AC mask')
   else:
    prev=previous[label]
    if a['excluded']!=prev['excluded'] or a['supervised']!=prev['retained'] or not near(am[0],prev['original_ac_mass']) or not near(am[2],prev['retained_ac_mass']):raise ValueError('actual epoch uses preceding completed NEXT mask/weights')
   if epoch<=3 and a['excluded']!=0:raise ValueError('three completed observations required before applied exclusion')
   original+=am[0];removed+=am[1];retained+=am[2];exclusions+=a['excluded'];supervised+=a['supervised'];next_exclusions+=n['excluded']
  if actual.get('excluded')!=exclusions or actual.get('supervised')!=supervised or supervised+exclusions!=len(ix) or nxt.get('excluded')!=next_exclusions:raise ValueError('global/class count consistency')
  if not near(actual.get('original_AC_mass'),original) or not near(actual.get('removed_AC_mass'),removed) or not near(actual.get('retained_AC_mass'),retained) or not near(actual.get('removed_AC_mass_fraction'),removed/original):raise ValueError('global/class applied mass consistency')
  summary.append(dict(epoch=epoch,applied=dict(actual),next=dict(nxt),skipped_optimizer_batches=detail['skipped_optimizer_batches']))
  grad=row.get('gradient_diagnostics',{});good=grad.get('amp_successful_optimizer_updates');skipped=grad.get('amp_skipped_optimizer_updates')
  if not integer(good) or good<0 or not integer(skipped) or skipped<0 or good+skipped+detail['skipped_optimizer_batches']!=math.ceil(len(ix)/64):raise ValueError('AMP plus all-masked skips account for every real batch')
  successful+=good
  previous=nr
 if successful<=0:raise ValueError('real optimizer updates required')
 for moment in state.get('optimizer',{}).get('state',{}).values():
  step=moment.get('step');step=float(step) if isinstance(step,torch.Tensor) else step
  if step!=successful:raise ValueError('PF successful updates equal exact AdamW moment steps')
 # Exact terminal keep/streak/AC state proves the next mask for epoch21, not epoch20.
 for label,n in enumerate(previous):
  mask=ys==label;original=float(gate.last_ac_weights[mask].double().sum());removed=float(gate.last_ac_weights[mask & ~gate.keep].double().sum())
  if n['excluded']!=int((mask & ~gate.keep).sum()) or not near(n['original_ac_mass'],original) or not near(n['retained_ac_mass'],original-removed):raise ValueError('terminal NEXT state/class mass exact restoration')
 return dict(schema='aictrack1.persistent_filter_joint.applied_supervision_audit.v1',complete=True,protocol=recipe['protocol'],max_ac_weight=.30,applied_epochs_checked=expected_epochs,training_count=len(ix),num_classes=num_classes,gate_AC_weights_equal=True,successful_optimizer_updates=successful,all_masked_batches_accounted=True,terminal_next_mask_sha256=hashlib.sha256(gate.keep.numpy().tobytes(order='C')).hexdigest(),terminal_next_excluded=int((~gate.keep).sum()),per_epoch=summary,definition='actual epoch snapshots checked against previous completed NEXT summaries; terminal next mask is not epoch20 supervision',historical_mask_sample_identities_reconstructed=False,noise_precision_claim=False,test_access=False)

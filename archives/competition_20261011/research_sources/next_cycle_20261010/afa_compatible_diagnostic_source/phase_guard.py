"""One actual root-authorized AFA <=4h ledger covers training and frozen validation."""
from __future__ import annotations
import json,math,os,shutil,sys,time
from pathlib import Path
from fresh_diag_contract import ROOT,REMOTE_ROOT,TRAIN_ROOT,TRAINING_SOURCE_SHA,sha,valid_digest,verify_training_binding,validate_checkpoint_metadata,validate_state,validate_optimizer_state,validate_rng_state,validate_ac_state,validate_afa_checkpoint,confirmation_rows
BUDGET_SHA=json.loads((ROOT/'inputs/diagnostic_binding.json').read_text(encoding='utf-8'))['budget_sha256']

def validate_budget_payload(b):
 expected=dict(schema='aictrack1.AFA_CLIP_anchored.budget.v1',initial_free_bytes=13013950464,capacity_increase_bytes=32212254720,adjusted_initial_free_bytes=45226205184,data_total_bytes=128849018880,disk_max_bytes=32212254720,min_free_bytes=3221225472,atomic_reserve_bytes=2147483648,final_gpu_reserve_hours=4,root_authorized=True,test_access=False)
 if any(b.get(k)!=v for k,v in expected.items()) or b['adjusted_initial_free_bytes']!=b['initial_free_bytes']+b['capacity_increase_bytes']:raise ValueError('original expanded cumulative ledger and AFA typed budget')
 if not all(isinstance(b.get(k),(int,float)) and not isinstance(b.get(k),bool) and math.isfinite(b[k]) for k in ('phase_start_unix','deadline_unix')) or not 0<b['deadline_unix']-b['phase_start_unix']<=14400:raise ValueError('new explicit finite shared <=4h clock required')
 evidence=b.get('capacity_evidence',{})
 if evidence!={'path':'/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_continuation_20261009/root_actual_disk_expansion_recovery_receipt.json','sha256':'ba819e4b72731e810743deb1b4b4849eb070e8c664187aeaf255ab3840a78e52'}:raise ValueError('original expansion provenance required')
 return b

def read_json(path,limit=2*1024**2):
 path=Path(path)
 if not path.is_file() or path.stat().st_size>limit:raise ValueError('bounded JSON file required '+str(path))
 return json.loads(path.read_text(encoding='utf-8'))

def phase_context(training_root,*,live=False):
 training_root=Path(training_root).resolve();a,m=verify_training_binding(training_root,TRAINING_SOURCE_SHA,runtime_assets=live)
 path=training_root/'inputs/budget.json'
 if sha(path)!=BUDGET_SHA or sha(ROOT/'inputs/budget.json')!=BUDGET_SHA:raise ValueError('same frozen training-plus-validation4h budget required')
 b=read_json(path)
 validate_budget_payload(b)
 if m.get('budget_sha256')!=BUDGET_SHA:raise ValueError('training/diagnostic exact same new budget')
 context=dict(budget=b,budget_path=str(path),budget_sha256=BUDGET_SHA,training_source_manifest_sha256=TRAINING_SOURCE_SHA,old_query_reused_for_exploration=True,independent_confirmation=False)
 if live:
  if os.name!='posix' or training_root!=TRAIN_ROOT.resolve() or ROOT.resolve()!=REMOTE_ROOT.resolve():raise ValueError('dedicated new producer/diagnostic roots required')
  sys.path[:0]=[str(training_root/'scripts'),str(training_root/'src')]
  e=b['capacity_evidence']
  if sha(e['path'])!=e['sha256']:raise ValueError('actual expansion receipt SHA')
  value=read_json(e['path'])
  if value.get('data_capacity_increase_bytes')!=b['capacity_increase_bytes'] or value.get('adjusted_initial_free_bytes')!=b['adjusted_initial_free_bytes'] or value.get('original_initial_free_bytes')!=b['initial_free_bytes']:raise ValueError('actual expansion accounting')
  resource_check(context)
 return context

def resource_check(context,reserve=0):
 b=context['budget'];disk=shutil.disk_usage(TRAIN_ROOT);now=time.time()
 if disk.total!=b['data_total_bytes'] or not b['phase_start_unix']<=now<b['deadline_unix'] or disk.free-reserve<b['min_free_bytes'] or b['adjusted_initial_free_bytes']-disk.free+reserve>b['disk_max_bytes']:raise RuntimeError('shared4h or exact capacity/cumulative ledger exceeded')
 return now

def terminal_pair_admission(training_root,launch_path,launch_sha):
 """Actual launch, every owned group closure, exact full20 payloads; never starts a process."""
 import torch,yaml
 training_root=Path(training_root).resolve();context=phase_context(training_root,live=True)
 reference=read_json(ROOT/'inputs/actual_training_launch_reference.json')
 if launch_sha!=reference['remote_sha256'] or Path(launch_path).as_posix()!=reference['remote_path'] or sha(launch_path)!=launch_sha:raise ValueError('actual controller launch explicit SHA')
 launch=read_json(launch_path)
 snapshot=ROOT/'inputs'/reference['local_snapshot']
 if sha(snapshot)!=reference['local_snapshot_sha256'] or launch!=read_json(snapshot):raise ValueError('actual controller launch object differs from frozen observed contents')
 controller=validate_actual_launch(launch,TRAINING_SOURCE_SHA)
 owner=normalize_owner(controller)
 if controller.get('cwd')!=TRAIN_ROOT.as_posix():raise ValueError('isolated actual AFA controller cwd')
 sys.path[:0]=[str(training_root/'scripts'),str(training_root/'src')]
 from owned_launch import snapshot,owned_group
 from aictrack1.engine.training_queue import read_training_lock,probe_running_trainers
 from p1_contract import assets,verify_remote_assets,validate_config
 values=snapshot()
 if owned_group(values,owner,[]) or read_training_lock(assets()['canonical_lock']) or probe_running_trainers():raise RuntimeError('original controller/group or canonical trainer still active')
 a=verify_remote_assets();_,parts=confirmation_rows();indices=[];targets=[]
 with Path(a['assets']['manifest']['path']).open(encoding='utf-8-sig',newline='') as handle:
  import csv
  rows=list(csv.DictReader(handle));index={row['sample_id']:i for i,row in enumerate(rows)}
 for row in parts['train']:indices.append(index[row['sample_id']]);targets.append(int(row['target']))
 receipts=[];checkpoints={}
 for arm in ('AFA',):
  attempts=[]
  for start in sorted((training_root/'artifacts/launches').glob(arm+'_*/START.json')):
   value=read_json(start)
   if '--p1-smoke' in value.get('command',[]):continue
   if value.get('source_manifest_sha256')!=TRAINING_SOURCE_SHA:raise ValueError('real arm launch source')
   if owned_group(values,normalize_owner(value['owner']),[]):raise RuntimeError('original owned arm group active')
   end=start.with_name('EXIT.json');terminal=read_json(end)
   if terminal.get('owner')!=value['owner'] or terminal.get('group_closed') is not True or (not isinstance(terminal.get('return_code'),int) or isinstance(terminal.get('return_code'),bool)):raise ValueError('actual arm closure receipt')
   attempts.append(dict(start_path=str(start),start_sha256=sha(start),exit_path=str(end),exit_sha256=sha(end),return_code=terminal['return_code'],owner=value['owner']))
  if not attempts or not any(x['return_code']==0 for x in attempts):raise ValueError('actual successful terminal arm launch required')
  config=training_root/f'configs/experiment/p1_{arm}.yaml';raw=yaml.safe_load(config.read_text(encoding='utf-8'))['experiment'];validate_config(raw,arm,a)
  last=training_root/f'artifacts/experiments/JOINT_{arm}/last.pt';metrics_path=last.with_name('metrics.json');metrics=read_json(metrics_path,32*1024**2);digest=sha(last)
  state=torch.load(last,map_location='cpu',weights_only=True,mmap=True)
  validate_checkpoint_metadata(state,metrics,raw,arm=arm,config_sha=sha(config),last_sha=digest,registered=dict(a,training_source_manifest_sha256=TRAINING_SOURCE_SHA))
  names=list(state['trainable_state']);visual=[n for n in names if n.startswith('visual.')]
  validate_rng_state(state);validate_state(state['trainable_state'],arm=arm);validate_optimizer_state(state,names,visual_names=visual,arm=arm);validate_ac_state(state['confidence_weighting'],indices,targets,raw['confidence_weighting'])
  validate_afa_checkpoint(state,metrics,raw,training_root=training_root)
  checkpoints[arm]=dict(path=str(last),sha256=digest,metrics_path=str(metrics_path),metrics_sha256=sha(metrics_path),epoch=20,config_sha256=sha(config));del state
  receipts.append(dict(arm=arm,launch_attempts=attempts))
 resource_check(context)
 return dict(actual=True,training_source_manifest_sha256=TRAINING_SOURCE_SHA,launch_path=str(Path(launch_path).resolve()),launch_sha256=launch_sha,controller_owner=owner,controller_and_all_arm_groups_closed=True,canonical_idle=True,checkpoints=checkpoints,arm_closure_evidence=receipts,shared_phase_context=context,test_access=False)


def normalize_owner(value):
 keys=('pid','pgrp','session')
 if any(not isinstance(value.get(k),int) or isinstance(value.get(k),bool) or value[k]<=0 for k in keys):raise ValueError('actual PID/session integer')
 start=value.get('start_ticks')
 if isinstance(start,bool) or not isinstance(start,(str,int)) or not str(start).isdigit() or int(start)<=0:raise ValueError('actual process birth ticks')
 owner={k:value[k] for k in keys};owner['start_ticks']=str(int(start))
 if owner['pid']!=owner['pgrp'] or owner['pid']!=owner['session']:raise ValueError('isolated actual owned session required')
 return owner

def validate_actual_launch(launch,source_sha,*,budget_sha=BUDGET_SHA):
 if launch.get('test_access') is not False:raise ValueError('AFA launch no test access')
 actual=[launch[k] for k in ('source_manifest_sha256','bound_source_sha256') if k in launch]
 if not actual or any(v!=source_sha for v in actual):raise ValueError('actual AFA launch source')
 if 'budget_sha256' in launch and launch['budget_sha256']!=budget_sha:raise ValueError('actual AFA launch same AFA budget')
 if 'initialize_from_smoke' in launch and launch['initialize_from_smoke'] is not False:raise ValueError('smoke not candidate initializer')
 if 'mode' in launch and launch['mode']!='train':raise ValueError('only actual train launch')
 if 'planned_arms' in launch and launch['planned_arms']!=['AFA']:raise ValueError('only scheduled AFA')
 if 'epochs_each' in launch and launch['epochs_each']!=20:raise ValueError('continuous20 only')
 if 'planned_pause' in launch and launch['planned_pause'] is not False:raise ValueError('no planned pause')
 controller=launch.get('controller',launch.get('process'))
 if not isinstance(controller,dict):raise ValueError('actual controller birth receipt')
 normalize_owner(controller)
 command=launch.get('command',controller.get('argv'))
 if command is None:command=controller.get('command')
 if not isinstance(command,list) or not command or any(not isinstance(x,str) for x in command):raise ValueError('actual AFA controller command')
 if any(x in command for x in ('--preflight','--smoke','--profile','--stop-after-epoch')):raise ValueError('preflight/smoke/partial is not production')
 if not any(Path(x).as_posix().endswith('scripts/run_supervision_joint.py') for x in command):raise ValueError('actual AFA canonical queue entry')
 return controller

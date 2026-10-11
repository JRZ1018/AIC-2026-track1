"""One existing root-authorized8h ledger covers training and frozen validation."""
from __future__ import annotations
import json,math,os,shutil,sys,time
from pathlib import Path
from fresh_diag_contract import ROOT,REMOTE_ROOT,TRAIN_ROOT,TRAINING_SOURCE_SHA,sha,valid_digest,verify_training_binding,validate_checkpoint_metadata,validate_state,validate_optimizer_state,validate_rng_state,validate_ac_state,confirmation_rows,validate_soft_state,BINDING
BUDGET_SHA=BINDING['budget_sha256']

def read_json(path,limit=2*1024**2):
 path=Path(path)
 if not path.is_file() or path.stat().st_size>limit:raise ValueError('bounded JSON file required '+str(path))
 return json.loads(path.read_text(encoding='utf-8'))

def phase_context(training_root,*,live=False):
 training_root=Path(training_root).resolve();a,m=verify_training_binding(training_root,TRAINING_SOURCE_SHA,runtime_assets=live)
 path=training_root/'inputs/budget.json'
 if sha(path)!=BUDGET_SHA or sha(ROOT/'inputs/budget.json')!=BUDGET_SHA or a['budget_sha256']!=BUDGET_SHA:raise ValueError('one actual shared H1+visual+diagnostic budget')
 from fresh_route_runtime import budget
 b=budget(training_root)
 context=dict(budget=b,budget_path=str(path),budget_sha256=BUDGET_SHA,training_source_manifest_sha256=TRAINING_SOURCE_SHA,independent_confirmation=True,test_access=False)
 if live:
  if os.name!='posix' or training_root!=TRAIN_ROOT.resolve() or ROOT.resolve()!=REMOTE_ROOT.resolve():raise ValueError('dedicated new610 training/diagnostic roots')
  resource_check(context)
 return context

def resource_check(context,reserve=0):
 b=context['budget'];disk=shutil.disk_usage(TRAIN_ROOT);now=time.time()
 if disk.total!=b['actual_disk_total_bytes'] or not b['phase_start_unix']<=now<b['deadline_unix'] or disk.free-reserve<b['min_free_bytes'] or b['adjusted_initial_free_bytes']-disk.free+reserve>b['disk_max_bytes']:raise RuntimeError('actual shared confirmation clock/cumulative ledger exceeded')
 return now

def normalize_owner(value):
 owner={k:value.get(k) for k in ('pid','pgrp','session','start_ticks')}
 if any(type(owner[k]) is not int or owner[k]<=0 for k in ('pid','pgrp','session')) or owner['pid']!=owner['pgrp'] or owner['pid']!=owner['session'] or not isinstance(owner['start_ticks'],str) or not owner['start_ticks'].isdigit() or str(int(owner['start_ticks']))!=owner['start_ticks'] or int(owner['start_ticks'])<=0:raise ValueError('exact positive typed isolated birth required')
 return owner

def actual_command_arms(command,training_root,selected_arm):
 if selected_arm not in ('B25','B50') or not isinstance(command,list) or command[:1]!=['/root/autodl-tmp/venvs/aictrack1/bin/python']:raise ValueError('actual prescribed controller Python and one selected B')
 index=2 if command[1:2]==['-B'] else 1
 if len(command)<=index or (Path(training_root)/command[index]).resolve()!=(Path(training_root)/'scripts/run_confirmation_visual.py').resolve():raise ValueError('actual new610 controller entrypoint')
 flags=command[index+1:]
 if flags:
  if flags[0]!='--resume-arms' or len(set(flags[1:]))!=len(flags[1:]) or any(a not in ('J5',selected_arm) for a in flags[1:]):raise ValueError('production only; preflight/smoke/unregistered arms forbidden')
 return ['J5',selected_arm]


def terminal_pair_admission(training_root,launch_path,launch_sha,selected_arm):
 """Actual launch, every owned group closure, exact full20 payloads; never starts a process."""
 import torch,yaml
 training_root=Path(training_root).resolve();context=phase_context(training_root,live=True)
 reference=read_json(ROOT/'inputs/actual_training_launch_reference.json')
 if launch_sha!=reference['remote_sha256'] or Path(launch_path).as_posix()!=reference['remote_path'] or sha(launch_path)!=launch_sha:raise ValueError('actual controller launch explicit SHA')
 launch=read_json(launch_path)
 snapshot=ROOT/'inputs'/reference['local_snapshot']
 if sha(snapshot)!=reference['local_snapshot_sha256'] or launch!=read_json(snapshot):raise ValueError('actual controller launch object differs from frozen observed contents')
 if launch.get('mode','train')!='train' or launch.get('bound_source_sha256')!=TRAINING_SOURCE_SHA or launch.get('test_access') is not False:raise ValueError('actual production B root operator launch')
 arms=actual_command_arms(launch.get('command'),training_root,BINDING['selected_candidate'])
 if selected_arm not in arms:raise ValueError('diagnostic arm not in actual pair')
 controller=launch['process'];owner=normalize_owner(controller)
 if controller.get('cwd')!=TRAIN_ROOT.as_posix() or owner['pid']!=owner['pgrp'] or owner['pid']!=owner['session'] or not str(owner['start_ticks']).isdigit():raise ValueError('isolated actual controller identity')
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
 for arm in arms:
  attempts=[]
  for start in sorted((training_root/'artifacts/launches').glob(arm+'_*/START.json')):
   value=read_json(start)
   if '--p1-smoke' in value.get('command',[]) or '--p1-profile' in value.get('command',[]):continue
   if value.get('source_manifest_sha256')!=TRAINING_SOURCE_SHA:raise ValueError('real arm launch source')
   normalize_owner(value['owner'])
   if owned_group(values,value['owner'],[]):raise RuntimeError('original owned arm group active')
   end=start.with_name('EXIT.json');terminal=read_json(end)
   if terminal.get('owner')!=value['owner'] or terminal.get('group_closed') is not True or not isinstance(terminal.get('return_code'),int):raise ValueError('actual arm closure receipt')
   attempts.append(dict(start_path=str(start),start_sha256=sha(start),exit_path=str(end),exit_sha256=sha(end),return_code=terminal['return_code'],owner=value['owner']))
  if not attempts or not any(x['return_code']==0 for x in attempts):raise ValueError('actual successful terminal arm launch required')
  config=training_root/f'configs/experiment/p1_{arm}.yaml';raw=yaml.safe_load(config.read_text(encoding='utf-8'))['experiment'];validate_config(raw,arm,a)
  last=training_root/f'artifacts/experiments/CONFIRM_{arm}/last.pt';metrics_path=last.with_name('metrics.json');metrics=read_json(metrics_path,16*1024**2);digest=sha(last)
  state=torch.load(last,map_location='cpu',weights_only=True,mmap=True)
  validate_checkpoint_metadata(state,metrics,raw,arm=arm,config_sha=sha(config),last_sha=digest,registered=dict(a,training_source_manifest_sha256=TRAINING_SOURCE_SHA))
  names=list(state['trainable_state']);visual=[n for n in names if n.startswith('visual.')]
  validate_rng_state(state);validate_state(state['trainable_state'],arm=arm);validate_optimizer_state(state,names,visual_names=visual,arm=arm);validate_ac_state(state['confidence_weighting'],indices,targets,raw['confidence_weighting']);soft_audit=validate_soft_state(state,raw,torch.bincount(torch.tensor(targets),minlength=750))
  checkpoints[arm]=dict(path=str(last),sha256=digest,metrics_path=str(metrics_path),metrics_sha256=sha(metrics_path),epoch=20,config_sha256=sha(config),soft_method_audit=soft_audit);del state
  receipts.append(dict(arm=arm,launch_attempts=attempts))
 resource_check(context)
 return dict(actual=True,training_source_manifest_sha256=TRAINING_SOURCE_SHA,launch_path=str(Path(launch_path).resolve()),launch_sha256=launch_sha,controller_owner=owner,controller_and_all_arm_groups_closed=True,canonical_idle=True,checkpoints=checkpoints,arm_closure_evidence=receipts,shared_phase_context=context,test_access=False)

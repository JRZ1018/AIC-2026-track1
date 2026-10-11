"""FV2: two new same-seed final checkpoints; no historical prediction reuse or test."""
import argparse,dataclasses,hashlib,importlib.util,json,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SERVER=Path('/root/AICtrack1_fullvisual_confirm_diagnostic_v6_20261008')
TRAIN=Path('/root/AICtrack1_fullvisual_confirm_v5_20261008')
SOURCE_SHA='0f19ecb1347578f5b3eee02f048cda939b1bf70382858987297418e504a2c097'
HEAD_SHA='45266a98329a59f708b12f394775640e032e50be8c44077a1037048f28c88753'
LOCK=Path('/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json')
OLD_DIAG=Path('/root/autodl-tmp/AICtrack1_n1_diagnostic_20261003/scripts/diagnose_stage2_tta.py')
DIAG_SHA='7fd977bb31678bdf1d9e1ef302f232fa99803deb0f4fcb90409e1494da68ad96'
FINAL_LOADER_SHA='4315a435dd7bdf3b9c3856d1b68120fb3994e0dfec32e33f9168543fb26e3439'

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_owned_sources():
 m=json.loads((ROOT/'sources.json').read_text())
 for n,h in m['files'].items():
  p=Path(n);assert not p.is_absolute() and '..' not in p.parts and sha(ROOT/p)==h,n
 return sha(ROOT/'sources.json')
def arm_paths(arm):
 assert arm in ('last8','fullvisual')
 name='control20' if arm=='last8' else 'fullvisual20'
 return TRAIN/'configs/experiment'/f'{name}.yaml',TRAIN/'artifacts/experiments'/name/'last.pt'
def bundle_directory(arm):
 assert arm in ('last8','fullvisual');return ROOT/'artifacts/diagnostics'/arm
def verify_training():
 assert sha(TRAIN/'mainline_sources_sha256.json')==SOURCE_SHA
 for n,h in json.loads((TRAIN/'mainline_sources_sha256.json').read_text())['files'].items():assert sha(TRAIN/n)==h,n
 assert sha(TRAIN/'artifacts/experiments/mainline-h1/last.pt')==HEAD_SHA
 h=json.loads((TRAIN/'artifacts/experiments/mainline-h1/h1_receipt.json').read_text())
 assert h['complete'] and h['epoch']==100 and h['training_seed']==20261008 and not h['validation_selection']
 r=json.loads((TRAIN/'controller_result.json').read_text())
 assert r['complete'] and r['status']=='FV2_paired_training_complete_new_diagnostics_pending' and r['test_access'] is False
 for arm in ('last8','fullvisual'):
  config,p=arm_paths(arm);name=p.parent.name;a=json.loads((TRAIN/f'arms/{name}_complete.json').read_text())
  assert a['complete'] and a['epoch']==20 and a['ac_epochs']==20 and a['last_sha256']==sha(p) and a['head_sha256']==HEAD_SHA
def resource_check():
 import shutil
 sys.path[:0]=[str(TRAIN/'scripts'),str(TRAIN/'src')]
 from mainline_guard import memory_snapshot,memory_reason
 s=json.loads((ROOT/'stage_receipt.json').read_text());free=shutil.disk_usage(ROOT).free
 assert time.time()<s['deadline_epoch'] and free>=s['min_free_bytes']
 assert s['initial_free_bytes']-free<=s['disk_peak_limit_bytes']
 assert s['parent_initial_free_bytes']-free<=s['parent_disk_peak_limit_bytes']
 assert memory_reason(memory_snapshot()) is None
def load_model(**kwargs):
 import torch,yaml
 from mainline_h1_contract import load_new_fold_head
 config=kwargs['config_path'];p=kwargs['checkpoint_path'];e=yaml.safe_load(config.read_text())['experiment'];s=torch.load(p,weights_only=True,map_location='cpu')
 assert config in [arm_paths(a)[0] for a in ('last8','fullvisual')]
 assert kwargs['checkpoint_role']=='final' and s['epoch']==20 and s['config_sha256']==sha(config)
 assert e['training']['seed']==20261008 and e['training']['visual_learning_rate']==1.75e-6
 assert s['token_readout']==e['token_readout'] and e['token_readout']['mode']=='none'
 assert sha(Path(e['head_checkpoint']))==HEAD_SHA and s.get('head_initializer_sha256',HEAD_SHA)==HEAD_SHA
 loader=ROOT/'scripts/fv2_single_model.py';assert sha(loader)==FINAL_LOADER_SHA
 name='aictrack1.models.fv2_audited_final_loader';spec=importlib.util.spec_from_file_location(name,loader);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 m.load_classifier_head_checkpoint=load_new_fold_head
 device=kwargs['device'];fitted=m.load_diagnostic_model(**dict(kwargs,device=torch.device('cpu')));fitted.model.to(device)
 return dataclasses.replace(fitted,device=device)
def audit(fitted,arm):
 import torch,random,numpy as np,math,yaml
 from aictrack1.methods.confidence import ClasswiseHistoryWeighting
 from aictrack1.data.contracts import load_dataset_config
 from aictrack1.data.indexed_images import read_indexed_images
 config,p=arm_paths(arm);s=torch.load(p,weights_only=True,map_location='cpu');metrics=json.loads((p.parent/'metrics.json').read_text());e=yaml.safe_load(config.read_text())['experiment']
 count,numel,mode=(96,56702976,'last_n_blocks') if arm=='last8' else (152,87849216,'full_visual')
 assert e['peft']==dict(mode=mode,lora_blocks=8) and s['mode']==mode
 assert metrics['history']==s['history'] and metrics['early_stop'] is None and metrics['train_count']==118958 and metrics['validation_count']==29737
 assert [r['epoch'] for r in s['history']]==list(range(1,21))
 assert sha(p)==metrics['last_checkpoint_sha256'] and sha(p.parent/'best.pt')==metrics['selected_checkpoint_sha256']
 best=torch.load(p.parent/'best.pt',weights_only=True,map_location='cpu')
 assert best['config_sha256']==sha(config) and best['mode']==mode and best['split_sha256']==s['split_sha256']
 assert len(best['trainable_state'])==count and all(torch.isfinite(v).all() for v in best['trainable_state'].values());del best
 params=dict(fitted.model.named_parameters());names=list(s['trainable_state']);assert len(names)==count and sum(s['trainable_state'][n].numel() for n in names)==numel
 assert all(n.startswith('visual.') for n in names)
 if arm=='last8':assert all(n.startswith('visual.transformer.resblocks.') and int(n.split('.')[3]) in range(4,12) for n in names)
 else:assert set(names)=={n for n in params if n.startswith('visual.')}
 assert all(torch.equal(params[n].detach(),s['trainable_state'][n]) for n in names)
 opt=torch.optim.AdamW([params[n] for n in names],lr=e['training']['visual_learning_rate'],weight_decay=.01);opt.load_state_dict(s['optimizer'])
 assert len(opt.state)==count and len(opt.param_groups)==1
 for p0,pid in zip([params[n] for n in names],s['optimizer']['param_groups'][0]['params']):
  v=opt.state[p0];stored=s['optimizer']['state'][pid]
  assert v['exp_avg'].shape==p0.shape and v['exp_avg_sq'].shape==p0.shape
  assert all(torch.isfinite(x).all() for x in v.values() if isinstance(x,torch.Tensor))
  assert set(v)==set(stored) and all(torch.equal(v[k].cpu(),stored[k].cpu()) if isinstance(v[k],torch.Tensor) else v[k]==stored[k] for k in v)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20);sched.load_state_dict(s['scheduler']);assert sched.last_epoch==20 and sched.T_max==20
 scaler=torch.amp.GradScaler('cpu');scaler.load_state_dict(s['scaler']);assert scaler.state_dict()==s['scaler'] and math.isfinite(s['scaler']['scale']) and s['scaler']['scale']>0
 saved=random.getstate();random.setstate(s['python_rng_state']);random.setstate(saved)
 saved=np.random.get_state();nr=s['numpy_rng_state'];np.random.set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),*nr[2:]));np.random.set_state(saved)
 saved=torch.get_rng_state();torch.set_rng_state(s['torch_rng_state']);torch.set_rng_state(saved)
 assert len(s['cuda_rng_state_all'])==1 and s['cuda_rng_state_all'][0].dtype==torch.uint8
 c=s['confidence_weighting'];ac=ClasswiseHistoryWeighting(c['sample_indices'],c['class_ids'],momentum=.8,history_mix=.5,min_weight=.02,magnitude_mix=.75);ac.load_state_dict(c)
 rows=read_indexed_images(load_dataset_config(config.parent/e['dataset_config']),source='train',manifest_path=Path(e['manifest']),split_path=Path(e['split']))
 rows=[r for r in rows if r.partition=='train'];assert len(rows)==118958 and ac.epochs_updated==20
 assert torch.equal(ac.sample_indices,torch.tensor([r.sample_index for r in rows])) and torch.equal(ac.class_ids,torch.tensor([r.internal_class_index for r in rows]))
 assert torch.isfinite(ac.weights).all() and torch.equal(ac.seen,torch.ones_like(ac.seen))
 assert all(r['confidence_weighting']['next_effective_quality']['decision']!='L2' and not r['bootstrap_curriculum']['active'] for r in s['history'])
 assert 'persistent_filter_state' not in s and 'bootstrap_curriculum_state' not in s and s.get('low_rank_visual_protocol') is None
 report=dict(arm=arm,last_sha256=sha(p),best_sha256=sha(p.parent/'best.pt'),best_verified_not_selected=True,trainable_tensors=count,trainable_numel=numel,optimizer_scheduler_scaler_ac_restored=True,exact_global_train_id_label_binding=True,formal_ac_epochs_updated=20,rng_cpu_restore_cuda_serialization=True,bitwise_resume_claim=False,new_h1_sha256=HEAD_SHA,training_seed=20261008,historical_control_reused=False,test_access=False)
 with (ROOT/f'audit_{arm}.json').open('x') as f:json.dump(report,f,indent=2)
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=('last8','fullvisual'),required=True);p.add_argument('--run',action='store_true');p.add_argument('--cpu-model-check',action='store_true');a=p.parse_args()
 assert ROOT.resolve()==SERVER and not (ROOT/'DataSetStage2/test').exists();verify_owned_sources();verify_training();resource_check()
 sys.path[:0]=[str(TRAIN/'scripts'),str(TRAIN/'src')]
 import torch
 torch.set_num_threads(2)
 from mainline_h1_contract import SPLIT_SHA,MAPPING_SHA
 config,checkpoint=arm_paths(a.arm);digest=sha(checkpoint);metrics=json.loads((checkpoint.parent/'metrics.json').read_text())
 if a.cpu_model_check:
  assert not a.run
  fitted=load_model(config_path=config,checkpoint_path=checkpoint,class_mapping_digest=MAPPING_SHA,num_classes=750,split_sha256=SPLIT_SHA,device=torch.device('cpu'),git_commit=str(metrics.get('git_commit','unknown')),checkpoint_role='final')
  with torch.inference_mode():v=fitted.model(torch.zeros(1,3,336,336))
  assert v.shape==(1,750) and torch.isfinite(v).all();audit(fitted,a.arm);resource_check();print(json.dumps(dict(cpu_model_pass=True,arm=a.arm,synthetic_optimizer_steps=0)));return 0
 assert a.run
 from aictrack1.engine.n1_guard import exclusive_idle
 from aictrack1.engine.training_queue import read_training_lock,probe_running_trainers
 assert not read_training_lock(LOCK) and not probe_running_trainers() and not exclusive_idle()
 assert sha(OLD_DIAG)==DIAG_SHA
 spec=importlib.util.spec_from_file_location('fv2_audited_fourview',OLD_DIAG);d=importlib.util.module_from_spec(spec);d.__file__=str(ROOT/'scripts/audited_fourview.py');exec(compile(OLD_DIAG.read_bytes(),str(OLD_DIAG),'exec'),d.__dict__)
 d.ROOT=ROOT;d.load_diagnostic_model=load_model
 stage=json.loads((ROOT/'stage_receipt.json').read_text());seconds=min(900,int(stage['deadline_epoch']-time.time()-660));assert seconds>=60
 def cutoff(*_):raise TimeoutError('FV2 diagnostic original deadline')
 signal.signal(signal.SIGALRM,cutoff);signal.alarm(seconds)
 sys.argv=[str(OLD_DIAG),'--experiment-config',str(config),'--checkpoint',str(checkpoint),'--expected-checkpoint-sha256',digest,'--checkpoint-role','final','--shared-lock-path',str(LOCK),'--output-dir',str(bundle_directory(a.arm)),'--batch-size','64','--workers','4','--four-view-scale384','--save-fused-logits','--hard-seconds',str(seconds)]
 try:rc=d.main();resource_check();return rc
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())

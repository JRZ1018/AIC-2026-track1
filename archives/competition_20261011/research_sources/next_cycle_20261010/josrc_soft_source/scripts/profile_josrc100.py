"""Real official-train CUDA profile only; virtual epoch6 is never a candidate initializer."""
from __future__ import annotations
import copy,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from p1_contract import ROOT,assets,sha,require
from fresh_route_runtime import resource_check,assert_route_head,runtime_values
from aictrack1.methods.josrc_soft import SoftTeacher,SupportLedger,combine_loss,validate_recipe
from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict
from aictrack1.engine.ema import optimizer_step_completed

def cpu(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 if isinstance(value,list):return [cpu(v) for v in value]
 if isinstance(value,tuple):return tuple(cpu(v) for v in value)
 return value

def rng_state():
 n=np.random.get_state();return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),torch=torch.random.get_rng_state().clone(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(v):
 random.setstate(v['python']);np.random.set_state(v['numpy']);torch.random.set_rng_state(v['torch'])
 if torch.cuda.is_available():torch.cuda.set_rng_state_all(v['cuda'])

def state(model,teacher,support,optimizer,scheduler,scaler):
 return dict(student=trainable_state_dict(model),teacher=teacher.state_dict(),support=support.state_dict(),optimizer=cpu(optimizer.state_dict()),scheduler=copy.deepcopy(scheduler.state_dict()),scaler=copy.deepcopy(scaler.state_dict()),rng=rng_state())

def restore(v,model,teacher,support,optimizer,scheduler,scaler):
 load_trainable_state_dict(model,v['student']);teacher.load_state_dict(v['teacher']);support.load_state_dict(v['support']);optimizer.load_state_dict(v['optimizer']);scheduler.load_state_dict(v['scheduler']);scaler.load_state_dict(v['scaler']);restore_rng(v['rng'])

def maximum_difference(a,b):
 if isinstance(a,torch.Tensor):
  require(isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype,'nextstep tensor identity')
  if a.dtype.is_floating_point:
   require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'nextstep finite')
   # Bound temporary memory even for CLIP attention/MLP matrices.
   x=a.detach().cpu().flatten();y=b.detach().cpu().flatten();return max((float((x[i:i+1048576]-y[i:i+1048576]).abs().max()) for i in range(0,len(x),1048576)),default=0.)
  require(torch.equal(a.cpu(),b.cpu()),'nextstep integer/RNG state');return 0.
 if isinstance(a,dict):
  require(isinstance(b,dict) and set(a)==set(b),'nextstep keys');return max((maximum_difference(a[k],b[k]) for k in a),default=0.)
 if isinstance(a,(list,tuple)):
  require(type(a)==type(b) and len(a)==len(b),'nextstep sequence');return max((maximum_difference(x,y) for x,y in zip(a,b)),default=0.)
 require(a==b,'nextstep scalar');return 0.

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as h:json.dump(value,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())

def run(raw,config_path):
 import train_n1_peft as original
 started=time.monotonic();payload=assets();soft=validate_recipe(raw['josrc_soft']);require(soft['soft_coefficient']==.25 and payload.get('profile_only') is True,'profile100 B25 dedicated source only')
 require(torch.cuda.is_available(),'actual CUDA required, no CPU fake profile');device=torch.device('cuda');seed=int(raw['training']['seed'])
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 output=(ROOT/os.environ['P1_PROFILE_DIR']).resolve();require(output.is_relative_to((ROOT/'artifacts/profiles').resolve()),'isolated profile output');output.mkdir(parents=True,exist_ok=True)
 require(not (output/'profile_receipt.json').exists(),'profile actual output already complete')
 base=config_path.parent;resolve=lambda value:original._resolve(base,str(value))
 config=original.load_dataset_config(resolve(raw['dataset_config']));manifest=resolve(raw['manifest']);split=resolve(raw['split'])
 records=original.read_indexed_images(config,source='train',manifest_path=manifest,split_path=split,partitions=['train'])
 counts=torch.bincount(torch.tensor([int(x.internal_class_index) for x in records]),minlength=750);require(len(records)==118958 and bool((counts>0).all()),'full official train-only profile source750 supports')
 transform=original.build_split_transforms(config.transforms);dataset=original.IndexedImageDataset(records,transform['train'],allow_truncated_images=config.allow_truncated_images,stochastic_seed=seed)
 dataset=original.with_observation_view(dataset,transform['validation'],original.observation_protocol(original.WEAK,seed));dataset.set_epoch(5)
 t=raw['training'];loader=original._loader(dataset,batch_size=t['batch_size'],workers=t['workers'],prefetch_factor=t['prefetch_factor'],persistent_workers=t['persistent_workers'],shuffle=True,seed=seed+5,device=device,sample_weights=None)
 weight,weight_sha=original._model_settings(resolve(raw['model_config']));clip,_,metadata=original.load_clip_vit_b32(weight,weight_sha,device);visual=clip.visual.float();del clip
 original.enable_interpolated_visual_resolution(visual,target_resolution=336);original.configure_visual_peft(visual,'full_visual',lora_blocks=8)
 head,h1=original.load_classifier_head_checkpoint(resolve(raw['head_checkpoint']),feature_dim=metadata.feature_dim,num_classes=750,expected_clip_weight_sha256=metadata.weight_sha256,expected_class_mapping_sha256=__import__('p1_contract').MAPPING_SHA,expected_split_sha256=sha(split))
 model=original.configure_readout(visual,head,normalize_features=h1['feature_cache_metadata']['normalized'],setting=raw['token_readout'],device=device)
 assert_route_head(model,head,{},raw);model.train();vparams=[x for x in visual.parameters() if x.requires_grad];hparams=[x for x in head.parameters() if x.requires_grad]
 optimizer=torch.optim.AdamW([{'params':vparams,'lr':t['visual_learning_rate']},{'params':hparams,'lr':t['head_learning_rate']}],weight_decay=t['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20);scaler=torch.amp.GradScaler('cuda')
 teacher=SoftTeacher(model,soft);support=SupportLedger(750);initial_head={k:x.detach().cpu().clone() for k,x in head.state_dict().items()};torch.cuda.reset_peak_memory_stats();successful=0;attempts=0;processed=[];losses=[];norms=[];soft_losses=[];first_batch=True;timing_start=None;restore_receipt=None
 def step(batch):
  images=batch['image'].to(device,non_blocking=True);weak=batch['confidence_observation_image'].to(device,non_blocking=True);labels=batch['label'].to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
  with torch.autocast(device_type='cuda',dtype=torch.float16):
   sw,tw=teacher.weak_views_before_strong(weak,5);logits=model(images)
   hard=original._weighted_base_loss(original._base_loss_per_sample(logits,labels,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.1,sce_beta=1.,sce_log_zero=-4.),None)
  loss,detail=combine_loss(hard,logits,sw,tw,labels,soft,5);require(bool(torch.isfinite(loss)),'actual profile finite loss');scaler.scale(loss).backward();scaler.unscale_(optimizer)
  norm=torch.nn.utils.clip_grad_norm_([*vparams,*hparams],t['gradient_clip_norm']);previous=scaler.get_scale();scaler.step(optimizer);scaler.update();ok=bool(optimizer_step_completed(scaler_enabled=scaler.is_enabled(),before=previous,after=scaler.get_scale()));teacher.update_after_optimizer(ok);support.observe(labels,detail)
  return dict(completed=ok,loss=float(loss.detach()),soft_loss=float(detail['soft_loss']),gradient_norm=float(norm.detach()),batch_size=len(labels))
 for batch in loader:
  resource_check(ROOT)
  if first_batch:torch.cuda.synchronize();timing_start=time.monotonic();first_batch=False
  if successful==99:
   checkpoint=state(model,teacher,support,optimizer,scheduler,scaler)
   checkpoint.update(schema='aictrack1.josrc_inspired.profile_checkpoint.v1',candidate=False,initializes_candidates=False,successful_updates=successful,attempted_updates=attempts,virtual_epoch_index=5,actual_warmup_completed=False,source_manifest_sha256=os.environ['P1_SOURCE_SHA'],config_sha256=sha(config_path),math_protocol_sha256=sha(ROOT/'inputs/soft_math_protocol.json'))
   checkpoint_path=output/'restore99.pt';original._save_atomic(checkpoint,checkpoint_path)
   before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True);before.record();result=step(batch);after.record();torch.cuda.synchronize()
   if result['completed']:
    expected=state(model,teacher,support,optimizer,scheduler,scaler);elapsed100=time.monotonic()-timing_start;step_seconds=before.elapsed_time(after)/1000
    actual=torch.load(checkpoint_path,map_location='cpu',weights_only=True);require(actual['source_manifest_sha256']==os.environ['P1_SOURCE_SHA'] and actual['config_sha256']==sha(config_path) and actual['math_protocol_sha256']==sha(ROOT/'inputs/soft_math_protocol.json') and actual['successful_updates']==99,'actual disk exact profile99/source/config/math')
    restore(actual,model,teacher,support,optimizer,scheduler,scaler);replay=step(batch);torch.cuda.synchronize();restored=state(model,teacher,support,optimizer,scheduler,scaler)
    difference=maximum_difference(expected,restored);lossdiff=abs(result['loss']-replay['loss']);require(replay['completed'] and difference<=1e-7 and lossdiff<=1e-6,'actual teacher/student/AdamW/scaler/RNG nextstep recovery')
    restore_receipt=dict(complete=True,checkpoint_sha256=sha(checkpoint_path),successful_updates_before=99,after=100,maximum_tensor_difference=difference,loss_difference=lossdiff,all_teacher_and_student_and_optimizer_and_support_and_RNG_checked=True,replay_is_extra_diagnostic_update=True,profile_checkpoint_is_not_candidate=True,real_batch_sample_ids=list(batch['sample_id']),one_profile_batch_cuda_seconds=step_seconds)
    del actual,expected,restored,checkpoint
   else:elapsed100=None
  else:result=step(batch)
  attempts+=1;successful+=int(result['completed']);processed.extend(list(batch['sample_id']));losses.append(result['loss']);norms.append(result['gradient_norm']);soft_losses.append(result['soft_loss'])
  if attempts%10==0:print(json.dumps(dict(stage='josrc_actual_train_profile100',successful_updates=successful,attempts=attempts,loss=result['loss'],teacher_updates=teacher.successful_updates,virtual_epoch=6,candidate=False)),flush=True)
  if successful==100:break
  require(attempts<300,'bounded100 successful profile attempts')
 require(successful==100 and restore_receipt is not None and teacher.successful_updates==100 and teacher.skipped_updates==attempts-100,'profile100 real updates and strict teacher count')
 require(any(not torch.equal(initial_head[k],x.detach().cpu()) for k,x in head.state_dict().items()),'actual learned classifier head updated')
 torch.cuda.synchronize();end=time.monotonic();hard_batches=math.ceil(len(records)/t['batch_size']);seconds_per_attempt=elapsed100/attempts
 report=dict(schema='aictrack1.josrc_inspired.actual_profile100.v1',complete=True,candidate=False,initializes_candidates=False,smoke_only=True,model='one student, parameter EMA .999',final_model='student_only',actual_train_image_forward=True,test_access=False,query_image_forward=False,profile_virtual_epoch=6,actual_warmup_completed=False,hard_AC_status='cold original warmup weights; trained epoch6 AC/history not fabricated',scheduler_policy='Cos20 created; no epoch completed and no scheduler epoch step',additional_no_grad_forwards_per_batch=2,source_manifest_sha256=os.environ['P1_SOURCE_SHA'],config_sha256=sha(config_path),math_protocol_sha256=sha(ROOT/'inputs/soft_math_protocol.json'),official_clip_sha256=metadata.weight_sha256,H1_sha256=sha(resolve(raw['head_checkpoint'])),training_count=len(records),full_train_class_counts=counts.tolist(),trainable_tensors=154,trainable_parameters=88233966,successful_updates=successful,attempted_updates=attempts,AMP_skips=attempts-successful,teacher_updates=teacher.successful_updates,head_update_verified=True,startup_seconds=timing_start-started,first100_wall_seconds=elapsed100,first100_includes_one_disk_atomic_checkpoint=True,profile_total_seconds=end-started,seconds_per_attempt_with_one_atomic_checkpoint=seconds_per_attempt,projected_20_epoch_active_every_epoch_upper_seconds=seconds_per_attempt*hard_batches*20,projected_active15_epoch_seconds=seconds_per_attempt*hard_batches*15,projection_excludes_validation_and_first5_hard_epochs_and_terminal_IO=True,peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(),peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved(),processed_rows=len(processed),unique_train_ids=len(set(processed)),processed_train_ids_sha256=__import__('hashlib').sha256('\n'.join(processed).encode()).hexdigest(),loss_min=min(losses),loss_max=max(losses),soft_loss_min=min(soft_losses),soft_loss_max=max(soft_losses),gradient_nonfinite_count=sum(not math.isfinite(x) for x in norms),support=support.summary(),nextstep_disk_recovery=restore_receipt,runtime=runtime_values(torch),actual_budget_sha256=sha(ROOT/'inputs/budget.json'))
 write(output/'profile_receipt.json',report);write(output/'metrics.json',dict(complete=True,smoke_only=True,candidate=False,training_scope='profile_only',selected_checkpoint='profile_receipt.json',successful_updates=100,source_manifest_sha256=os.environ['P1_SOURCE_SHA']));print(json.dumps(dict(profile_complete=True,successful_updates=100,candidate=False,profile_receipt_sha256=sha(output/'profile_receipt.json'),seconds_per_attempt=seconds_per_attempt)),flush=True);return 0

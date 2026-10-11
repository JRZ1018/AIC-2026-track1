"""AFA-specific165/model-method/optimizer contracts; no model/checkpoint I/O in helpers."""
from __future__ import annotations
import inspect,json,math,sys
from pathlib import Path
import numpy as np
import torch
from view_features import FourViewFeatureCollector
AFA_NUMEL=88297579
AFA_KEYS=165
AFA_FEATURE_PROTOCOL='L2_normalize(mean(L2_normalize(enhanced_CLS_shared_head_input_each_of_four_views))) in float32 on CPU'

def scientific_module(training_root):
 training_root=Path(training_root).resolve();sys.path[:0]=[str(training_root/'src'),str(training_root/'scripts')]
 from aictrack1.models import afa_compatible as module
 if not Path(inspect.getfile(module)).resolve().is_relative_to(training_root):raise ValueError('actual AFA module outside exact producer')
 return module

def extra_shapes(training_root):
 module=scientific_module(training_root)
 model=module.FeatureDecomposition()
 return {'afa.'+name:tuple(p.shape) for name,p in model.named_parameters()}

def expected_afa_recipe(training_root):return dict(scientific_module(training_root).RECIPE)

def check_support_vector(value,name):
 if not isinstance(value,torch.Tensor) or value.dtype!=torch.long or tuple(value.shape)!=(750,) or bool((value<0).any()):raise ValueError('AFA exact support '+name)
 return value.cpu()

def validate_method(state,experiment,*,training_root,train_counts):
 module=scientific_module(training_root);recipe=module.RECIPE
 if experiment.get('afa_compatible')!=recipe:raise ValueError('exact fixed AFA recipe')
 from fresh_diag_contract import sha
 protocol=Path(training_root)/'inputs/afa_protocol.json'
 if state.get('afa_protocol_sha256')!=sha(protocol):raise ValueError('AFA method mathematical protocol SHA')
 value=state.get('afa_training_state',{})
 if value.get('schema')!='aictrack1.afa_CLIP_anchored.training_state.v1' or value.get('recipe')!=recipe or value.get('epoch_index')!=19 or value.get('generation_disabled_at_inference') is not True:raise ValueError('AFA exact final epoch19 method state')
 counts=check_support_vector(value.get('frequency_counts'),'frequency_counts');expected=torch.as_tensor(train_counts,dtype=torch.long)
 if not torch.equal(counts,expected) or bool((counts<=0).any()) or int(counts.sum())!=118958:raise ValueError('actual all750 train-only frequency')
 rng=value.get('generator_state')
 if not isinstance(rng,torch.Tensor) or rng.dtype!=torch.uint8 or rng.ndim!=1 or not rng.numel():raise ValueError('AFA private CPU generator byte RNG')
 torch.Generator(device='cpu').set_state(rng.cpu())
 current={k:check_support_vector(value.get(k),k) for k in ('observed_support','eligible_support','generated_support')}
 history=state.get('history')
 if not isinstance(history,list) or [r.get('epoch') for r in history]!=list(range(1,21)):raise ValueError('AFA complete20 history')
 previous={k:torch.zeros(750,dtype=torch.long) for k in current};active_steps=0;total_steps=0;skips=0
 batches=math.ceil(118958/64)
 for epoch,row in enumerate(history,1):
  d=row.get('afa_compatible',{});arrays={}
  for k in current:
   v=d.get(k)
   if not isinstance(v,list) or len(v)!=750 or any(type(x) is not int or x<0 for x in v):raise ValueError('actual AFA epoch support vector '+k)
   arrays[k]=torch.tensor(v,dtype=torch.long)
   if bool((arrays[k]<previous[k]).any()):raise ValueError('AFA support cumulative monotonicity')
  if not torch.equal(arrays['observed_support'],counts*epoch) or bool((arrays['eligible_support']>arrays['observed_support']).any()):raise ValueError('all real train supports preserved')
  generated=arrays['generated_support']-previous['generated_support'];eligible=arrays['eligible_support']-previous['eligible_support']
  if d.get('epoch_generated')!=int(generated.sum()) or d.get('epoch_trust_eligible')!=int(eligible.sum()) or bool(((generated>0)&(eligible==0)).any()):raise ValueError('generated donors actual eligible class supports')
  if int(generated.sum())>118958//2 or int(eligible.sum())>118958 or d.get('extra_visual_forward_calls')!=0 or d.get('class_residual_branch') is not True or d.get('anchored_variation_branch') is not True:raise ValueError('bounded same-forward AFA branch identity')
  if epoch<=10 and (int(generated.sum()) or int(eligible.sum())):raise ValueError('exact warmup without decomposition/synthesis')
  gain=d.get('residual_gain')
  if isinstance(gain,bool) or not isinstance(gain,(int,float)) or not math.isfinite(gain) or abs(gain)>.100001:raise ValueError('bounded finite real residual gain')
  gd=row.get('gradient_diagnostics',{});ok=gd.get('amp_successful_optimizer_updates');skip=gd.get('amp_skipped_optimizer_updates')
  if type(ok) is not int or type(skip) is not int or ok<0 or skip<0 or ok+skip!=batches:raise ValueError('actual full1859 AMP successful/skipped steps')
  total_steps+=ok;skips+=skip
  if epoch>10:active_steps+=ok
  previous=arrays
 if any(not torch.equal(current[k],previous[k]) for k in current):raise ValueError('checkpoint vs actual cumulative AFA support history')
 if total_steps<=0 or active_steps<=0:raise ValueError('actual successful original and active AFA updates required')
 return dict(afa_method_state_complete=True,afa_protocol_sha256=state['afa_protocol_sha256'],afa_recipe=dict(recipe),afa_epoch_index=19,afa_generation_disabled_at_inference=True,afa_private_generator_restored=True,afa_train_frequency_bound=True,afa_all_train_support_preserved=True,afa_successful_original_updates=total_steps,afa_successful_active_updates=active_steps,amp_skipped_updates=skips,afa_generated_total=int(current['generated_support'].sum()),afa_eligible_total=int(current['eligible_support'].sum()),model_role='single_enhanced_CLS_generation_disabled',original_AFA_reproduction=False)

def validate_AFA_optimizer(state,names,*,visual_names,training_root,train_counts):
 audit=validate_method(state,dict(afa_compatible=expected_afa_recipe(training_root)),training_root=training_root,train_counts=train_counts)
 shape=extra_shapes(training_root);extra=[n for n in names if n.startswith('afa.') and n!='afa.gain'];heads=['classifier.linear.weight','classifier.linear.bias']
 ordered=[n for n in names if n in set(visual_names)]+[n for n in names if n in heads]+extra+['afa.gain']
 optimizer=state.get('optimizer',{});groups=optimizer.get('param_groups',[]);counts=[len(visual_names),2,10,1];lrs=[1.75e-6,5e-6,1e-4,1e-4];decays=[.01,.01,.01,0.]
 if [len(g['params']) for g in groups]!=counts or set(extra)|{'afa.gain'}!=set(shape) or len(ordered)!=len(names):raise ValueError('exact AFA visual/head/decomposition/gain groups')
 for g,lr,decay in zip(groups,lrs,decays,strict=True):
  if tuple(g.get('betas',()))!=(.9,.999) or g.get('eps')!=1e-8 or g.get('weight_decay')!=decay or g.get('initial_lr')!=lr or not isinstance(g.get('lr'),(float,int)) or abs(g['lr'])>1e-12:raise ValueError('exact AFA AdamW/Cos20 group hyperparameters')
 sched=state.get('scheduler',{})
 if sched.get('base_lrs')!=lrs or sched.get('T_max')!=20 or sched.get('last_epoch')!=20:raise ValueError('four-group continuous20 scheduler')
 ids=[i for g in groups for i in g['params']]
 if len(set(ids))!=len(ids) or len(optimizer.get('state',{}))!=len(names):raise ValueError('AFA complete optimizer parameter IDs')
 for name,pid in zip(ordered,ids,strict=True):
  record=optimizer['state'].get(pid,{})
  for field in ('exp_avg','exp_avg_sq'):
   value=record.get(field)
   if not isinstance(value,torch.Tensor) or value.shape!=state['trainable_state'][name].shape or not bool(torch.isfinite(value).all()):raise ValueError('AFA optimizer finite moments '+name)
  step=record.get('step');step=float(step) if isinstance(step,torch.Tensor) else step
  expected=audit['afa_successful_active_updates'] if name.startswith('afa.') else audit['afa_successful_original_updates']
  if type(step) not in (int,float) or not math.isfinite(step) or step!=expected:raise ValueError('AFA actual moment step timing '+name)
 scale=state.get('scaler',{}).get('scale')
 if type(scale) not in (float,int) or not math.isfinite(scale) or scale<=0 or state.get('confidence_weighting',{}).get('epochs_updated')!=20:raise ValueError('AFA scaler/AC complete20')
 return dict(optimizer_parameter_groups=counts,optimizer_moments_finite=True,scheduler_epoch=20,ac_observation_epochs=20,afa_original_moment_steps=audit['afa_successful_original_updates'],afa_module_moment_steps=audit['afa_successful_active_updates'])

def validate_restored_method(model,state):
 actual=model.method_state_dict();expected=state['afa_training_state']
 if set(actual)!=set(expected) or any((not torch.equal(actual[k],v)) if isinstance(v,torch.Tensor) else actual[k]!=v for k,v in expected.items()):raise ValueError('actual permanent AFA method differs from checkpoint')
 if model.afa_epoch_index!=19 or model.afa_cache is not None or model.afa_enabled is not True:raise ValueError('actual epoch19 enhancement and generation-disabled inference')
 return dict(afa_method_restored_equal=True,afa_generation_invoked=False)


class EnhancedCLSFeatureCollector(FourViewFeatureCollector):
 """Observe shared-head input and count original visual calls; no extra model call."""
 def __init__(self,model):
  self.feature_dim=512;self.layout='four_calls_B';self.pending=[];self.chunks=[];self.calls=0;self.visual_calls=0
  self.handle=model.classifier.register_forward_pre_hook(self._head_capture)
  self.visual_handle=model.visual.register_forward_hook(self._visual_capture)
 def _head_capture(self,module,args):return self._capture(module,args,args[0])
 def _visual_capture(self,module,args,output):self.visual_calls+=1;return None
 def finish(self,expected_rows):
  if self.visual_calls!=self.calls:raise ValueError('exact one visual per enhanced CLS inference head input')
  return super().finish(expected_rows)
 def close(self):super().close();self.visual_handle.remove()

"""JoSRC-inspired train-only auxiliary soft repair, not a complete paper reproduction."""
from __future__ import annotations
import math
from typing import Mapping
import torch
from torch import nn
from aictrack1.engine.ema import TrainableEMA
from aictrack1.methods.confidence_observation import weak_observation_logits
VERSION='aictrack1.josrc_inspired.double_student_view.soft.v1'

def validate_recipe(raw):
 expected=dict(protocol=VERSION,enabled=True,ema_decay=.999,warmup_epochs=5,soft_coefficient=raw.get('soft_coefficient'),temperature=1.,target='detached_EMA_weak_probabilities',reliability='mean_JS(student_strong,hard_onehot;student_weak,hard_onehot)/ln2',stability='1-JS(student_strong,student_weak)/ln2',weight='noise*stability*EMA_max_probability',soft_normalization='batch_mean',soft_uses_AC=False,hard_sample_filter=False,final_model='student_only')
 if raw!=expected or raw.get('soft_coefficient') not in (0.,.25,.5):raise ValueError('unregistered JoSRC-inspired soft recipe')
 return expected

def recipe(coefficient):
 return validate_recipe(dict(protocol=VERSION,enabled=True,ema_decay=.999,warmup_epochs=5,soft_coefficient=coefficient,temperature=1.,target='detached_EMA_weak_probabilities',reliability='mean_JS(student_strong,hard_onehot;student_weak,hard_onehot)/ln2',stability='1-JS(student_strong,student_weak)/ln2',weight='noise*stability*EMA_max_probability',soft_normalization='batch_mean',soft_uses_AC=False,hard_sample_filter=False,final_model='student_only'))

def js_divergence(p,q):
 if p.ndim!=2 or q.shape!=p.shape or p.dtype!=torch.float32 or q.dtype!=torch.float32 or not bool(torch.isfinite(p).all() and torch.isfinite(q).all()) or bool((p<0).any() or (q<0).any()):raise ValueError('finite nonnegative FP32 probability matrices required')
 if not torch.allclose(p.sum(1),torch.ones(p.shape[0],device=p.device),rtol=0,atol=2e-5) or not torch.allclose(q.sum(1),torch.ones(q.shape[0],device=q.device),rtol=0,atol=2e-5):raise ValueError('normalized probabilities required')
 mid=(p+q)*.5;eps=torch.finfo(torch.float32).tiny
 value=.5*((p*(p.clamp_min(eps).log()-mid.clamp_min(eps).log())).sum(1)+(q*(q.clamp_min(eps).log()-mid.clamp_min(eps).log())).sum(1))
 if not bool(torch.isfinite(value).all()) or bool((value<-2e-6).any() or (value>math.log(2.)+2e-6).any()):raise ValueError('JS outside finite information bounds')
 return value.clamp(0,math.log(2.))

class SoftTeacher:
 def __init__(self,student:nn.Module,raw:Mapping):
  self.recipe=validate_recipe(dict(raw));self.ema=TrainableEMA(student,decay=self.recipe['ema_decay']);self.student=student;self.successful_updates=0;self.skipped_updates=0
 def active(self,epoch_index):return self.recipe['soft_coefficient']>0 and epoch_index>=self.recipe['warmup_epochs']
 def weak_views_before_strong(self,images,epoch_index):
  """Both weak views occur before strong autograd graph; RNG/modes restored by original helper."""
  if not self.active(epoch_index):return None,None
  student_weak=weak_observation_logits(self.student,images)
  with self.ema.applied():teacher_weak=weak_observation_logits(self.student,images)
  return student_weak,teacher_weak
 def update_after_optimizer(self,completed):
  if not isinstance(completed,bool):raise ValueError('actual GradScaler successful-step boolean required')
  if completed:self.ema.update();self.successful_updates+=1
  else:self.skipped_updates+=1
  if self.ema.updates!=self.successful_updates:raise RuntimeError('EMA must follow only successful optimizer steps')
 def state_dict(self):return dict(schema=VERSION,recipe=self.recipe,ema=self.ema.state_dict(),successful_updates=self.successful_updates,skipped_updates=self.skipped_updates,final_model='student_only')
 def load_state_dict(self,state):
  if state.get('schema')!=VERSION or state.get('recipe')!=self.recipe or state.get('final_model')!='student_only' or not isinstance(state.get('successful_updates'),int) or state['successful_updates']<0 or not isinstance(state.get('skipped_updates'),int) or state['skipped_updates']<0 or state['ema'].get('updates')!=state['successful_updates']:raise ValueError('exact teacher recipe/count restoration required')
  for name,parameter in self.ema.parameters.items():
   value=state['ema']['shadow'].get(name)
   if not isinstance(value,torch.Tensor) or value.dtype!=parameter.dtype or value.shape!=parameter.shape or not bool(torch.isfinite(value).all()):raise ValueError('exact complete finite EMA master state '+name)
  if set(state['ema']['shadow'])!=set(self.ema.parameters):raise ValueError('teacher complete parameter identity')
  self.ema.load_state_dict(state['ema']);self.successful_updates=state['successful_updates'];self.skipped_updates=state['skipped_updates']

def combine_loss(hard_loss,strong_logits,student_weak,teacher_weak,labels,raw,epoch_index):
 r=validate_recipe(dict(raw))
 if r['soft_coefficient']==0 or epoch_index<r['warmup_epochs']:
  return hard_loss,dict(active=False,soft_coefficient=r['soft_coefficient'],hard_branch_unchanged=True,soft_weights=torch.zeros_like(labels,dtype=torch.float32),teacher_predictions=labels.detach().clone(),soft_loss=hard_loss.detach()*0.)
 if student_weak is None or teacher_weak is None or student_weak.shape!=strong_logits.shape or teacher_weak.shape!=strong_logits.shape:raise ValueError('true same-student two views and separate EMA target required')
 if labels.ndim!=1 or labels.shape[0]!=strong_logits.shape[0] or labels.dtype!=torch.int64 or bool((labels<0).any() or (labels>=strong_logits.shape[1]).any()):raise ValueError('original hard labels required')
 strong=torch.softmax(strong_logits.detach().float(),1);weak=torch.softmax(student_weak.detach().float(),1);target=torch.softmax(teacher_weak.detach().float(),1)
 hard=torch.nn.functional.one_hot(labels,num_classes=strong.shape[1]).float();noise=(js_divergence(strong,hard)+js_divergence(weak,hard))/(2*math.log(2.));stability=1-js_divergence(strong,weak)/math.log(2.);weight=(noise*stability*target.max(1).values).clamp(0,1).detach()
 per=-(target*torch.log_softmax(strong_logits.float(),1)).sum(1);soft=(weight*per).mean()
 if not bool(torch.isfinite(soft)):raise FloatingPointError('nonfinite auxiliary soft branch')
 return hard_loss+r['soft_coefficient']*soft,dict(active=True,soft_coefficient=r['soft_coefficient'],hard_branch_unchanged=True,soft_weights=weight,teacher_predictions=target.argmax(1),soft_loss=soft.detach(),noise=noise.detach(),stability=stability.detach(),teacher_confidence=target.max(1).values.detach())

class SupportLedger:
 def __init__(self,num_classes):
  self.num_classes=num_classes;self.hard_seen=torch.zeros(num_classes,dtype=torch.int64);self.soft_mass=torch.zeros(num_classes,dtype=torch.float64);self.teacher_transitions=torch.zeros(num_classes,num_classes,dtype=torch.int64);self.active_teacher_observations=0
 def observe(self,labels,detail):
  y=labels.detach().cpu().long();w=detail['soft_weights'].detach().cpu().double();p=detail['teacher_predictions'].detach().cpu().long()
  if y.shape!=w.shape or p.shape!=y.shape or not bool(torch.isfinite(w).all()) or bool((w<0).any()) or bool((y<0).any() or (y>=self.num_classes).any() or (p<0).any() or (p>=self.num_classes).any()):raise ValueError('support original-label binding')
  self.hard_seen+=torch.bincount(y,minlength=self.num_classes);self.soft_mass.index_add_(0,y,w)
  if detail.get('active') is True:
   self.teacher_transitions+=torch.bincount(y*self.num_classes+p,minlength=self.num_classes**2).reshape(self.num_classes,self.num_classes);self.active_teacher_observations+=len(y)
 def state_dict(self):return dict(num_classes=self.num_classes,hard_seen=self.hard_seen.clone(),soft_mass=self.soft_mass.clone(),teacher_transitions=self.teacher_transitions.clone(),active_teacher_observations=self.active_teacher_observations,hard_branch_dropped_samples=0,soft_only_additive=True)
 def load_state_dict(self,v):
  if v.get('num_classes')!=self.num_classes or v.get('hard_branch_dropped_samples')!=0 or v.get('soft_only_additive') is not True:raise ValueError('support protection schema')
  for name,dtype,shape in [('hard_seen',torch.int64,(self.num_classes,)),('soft_mass',torch.float64,(self.num_classes,)),('teacher_transitions',torch.int64,(self.num_classes,self.num_classes))]:
   x=v[name]
   if not isinstance(x,torch.Tensor) or x.dtype!=dtype or tuple(x.shape)!=shape or not bool(torch.isfinite(x).all()) or bool((x<0).any()):raise ValueError('support payload '+name)
   setattr(self,name,x.clone())
  self.active_teacher_observations=v['active_teacher_observations']
  if not isinstance(self.active_teacher_observations,int) or self.active_teacher_observations<0 or int(self.teacher_transitions.sum())!=self.active_teacher_observations:raise ValueError('only actual active teacher observations counted')
 def summary(self):return dict(hard_seen=self.hard_seen.tolist(),soft_mass=self.soft_mass.tolist(),hard_unseen_classes=(self.hard_seen==0).nonzero().flatten().tolist(),hard_branch_dropped_samples=0,soft_only_additive=True,teacher_transitions_total=int(self.teacher_transitions.sum()),active_teacher_observations=self.active_teacher_observations)

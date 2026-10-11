"""Formula-based CLIP migration of AFA: decomposition and controlled variation synthesis.
Not a reproduction of original ResNeXt/SFG/LBLS recipe. No third-party code copied.
"""
from __future__ import annotations
import copy,math
import torch
import torch.nn.functional as F
from torch import nn
from .token_attention_classifier import TokenAttentionClassifier,projected_patch_tokens
RECIPE=dict(protocol='AFA-CLIP-anchored-v1',feature_dim=512,patch_count=100,initialization_seed=20261008,warmup_epochs=10,auxiliary_coefficient=.25,residual_cap=.10,generated_shift=.10,generated_ratio=.5,frequency_power=1.,trusted_ac_min=.5,require_observed_label_prediction=True,spatial_V_initialization='identity',mask_axis='channel',generation_uses='official_fold_train_only',final_model='single_enhanced_CLS_generation_disabled')

def validate_recipe(recipe):
 if recipe not in (RECIPE,dict(RECIPE,auxiliary_coefficient=0.)):raise ValueError('fixed recipe or zero-coefficient CPU equivalence fixture')

def channel_decomposition(x,q,k,v,se):
 # x: B,C,L; Eq1 uses channel attention scaled by spatial extent L.
 if x.ndim!=3:raise ValueError('patch feature axes')
 query,key,value=q(x),k(x),v(x)
 attended=torch.softmax((query@key.transpose(1,2))/math.sqrt(x.shape[-1]),dim=-1)@value
 weights=torch.softmax(attended+se(attended),dim=1)
 class_features=weights*value
 return class_features,value-class_features,value,weights

class FeatureDecomposition(nn.Module):
 def __init__(self,feature_dim=512,patch_count=100,seed=20261008):
  super().__init__()
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):
   torch.manual_seed(seed)
   self.query=nn.Linear(patch_count,patch_count);self.key=nn.Linear(patch_count,patch_count)
   self.value=nn.Linear(patch_count,patch_count)
   with torch.no_grad():self.value.weight.copy_(torch.eye(patch_count));self.value.bias.zero_()
   self.se_down=nn.Linear(feature_dim,max(1,feature_dim//16));self.se_up=nn.Linear(max(1,feature_dim//16),feature_dim)
   self.gain=nn.Parameter(torch.zeros(()))
 def se(self,x):
  gate=torch.sigmoid(self.se_up(F.relu(self.se_down(x.mean(dim=-1)))))
  return x*gate.unsqueeze(-1)
 def forward(self,tokens):
  with torch.autocast(device_type=tokens.device.type,enabled=False):
   return channel_decomposition(tokens.float().transpose(1,2),self.query,self.key,self.value,self.se)

class AFACompatibleClassifier(TokenAttentionClassifier):
 def __init__(self,visual,classifier,*,normalize_features,recipe=RECIPE):
  validate_recipe(recipe)
  super().__init__(visual,classifier,normalize_features=normalize_features,mode='none')
  self.afa=FeatureDecomposition();self.afa_recipe=copy.deepcopy(recipe);self.afa_epoch_index=0
  self.afa_generator=torch.Generator(device='cpu');self.afa_generator.manual_seed(recipe['initialization_seed'])
  self.afa_class_counts=None;self.afa_cache=None;self.afa_enabled=True
  self.afa_eligible_support=torch.zeros(750,dtype=torch.long);self.afa_observed_support=torch.zeros(750,dtype=torch.long);self.afa_generated_support=torch.zeros(750,dtype=torch.long)
 def set_afa_training_context(self,epoch_index,class_counts):
  if type(epoch_index) is not int or not 0<=epoch_index<20:raise ValueError('AFA continuous20 epoch')
  counts=class_counts.detach().to(device='cpu',dtype=torch.long)
  if counts.shape!=(750,) or bool((counts<=0).any()):raise ValueError('all750 train-only frequency supports')
  if self.afa_class_counts is not None and not torch.equal(self.afa_class_counts,counts):raise ValueError('AFA frequency source changed')
  self.afa_class_counts=counts.clone();self.afa_epoch_index=epoch_index
 def classify_features(self,x):
  return self.classifier(F.normalize(x.float(),dim=-1) if self.normalize_features else x.float())
 def forward(self,images):
  self.afa_cache=None
  if not self.afa_enabled or self.afa_recipe['auxiliary_coefficient']==0 or self.afa_epoch_index<self.afa_recipe['warmup_epochs']:
   return super().forward(images)
  global_features,tokens=projected_patch_tokens(self.visual,images)
  class_features,variations,value,_=self.afa(tokens)
  enhanced=global_features+self.afa_recipe['residual_cap']*torch.tanh(self.afa.gain)*class_features.mean(dim=-1)
  if self.training:self.afa_cache=(global_features,class_features,variations,value)
  return self.classify_features(enhanced)
 def auxiliary_loss(self,labels,logits,ac_weights,*,coefficient=None):
  coef=self.afa_recipe['auxiliary_coefficient'] if coefficient is None else float(coefficient)
  if not math.isfinite(coef) or coef<0:raise ValueError('finite nonnegative auxiliary coefficient')
  detail=dict(active=False,generated_count=0,trust_eligible_count=0,generated_per_class=[0]*750)
  if self.training:self.afa_observed_support+=torch.bincount(labels.detach().cpu().long(),minlength=750)
  if coef==0 or self.afa_cache is None or not self.training:return logits.new_zeros(()),detail
  if self.afa_class_counts is None:raise ValueError('train-only class frequency required')
  batch=len(labels);weights=torch.ones(batch,device=labels.device) if ac_weights is None else ac_weights.detach()
  if weights.shape!=(batch,) or not bool(torch.isfinite(weights).all()):raise ValueError('actual applied hard AC weights')
  y=labels.detach().cpu().long();trusted=((weights>=self.afa_recipe['trusted_ac_min'])&(logits.detach().argmax(1)==labels)).cpu()
  self.afa_eligible_support+=torch.bincount(y[trusted],minlength=750)
  classes=torch.unique(y[trusted],sorted=True);limit=int(math.floor(batch*self.afa_recipe['generated_ratio']))
  detail.update(active=True,trust_eligible_count=int(trusted.sum()))
  if batch<2 or limit<1 or len(classes)==0:return logits.new_zeros(()),detail
  probabilities=self.afa_class_counts[classes].double().pow(-self.afa_recipe['frequency_power']);probabilities/=probabilities.sum()
  selected_classes=classes[torch.multinomial(probabilities,limit,replacement=True,generator=self.afa_generator)]
  class_indices=[];variation_indices=[]
  for cls in selected_classes:
   choices=torch.where(trusted&(y==cls))[0];i=int(choices[torch.randint(len(choices),(1,),generator=self.afa_generator)])
   j=int(torch.randint(batch-1,(1,),generator=self.afa_generator));j+=int(j>=i)
   class_indices.append(i);variation_indices.append(j)
  i=torch.tensor(class_indices,device=labels.device);j=torch.tensor(variation_indices,device=labels.device)
  global_features,class_features,variation,value=self.afa_cache
  # Eq11 is computed; CLS anchor explicitly differs from original AFA.
  zgen=(class_features.index_select(0,i)+variation.index_select(0,j)).mean(dim=-1)
  generated=global_features.index_select(0,i)+self.afa_recipe['generated_shift']*(zgen-value.index_select(0,i).mean(dim=-1))
  generated_labels=labels.index_select(0,i)
  loss=coef*F.cross_entropy(self.classify_features(generated).float(),generated_labels)
  per_class=torch.bincount(generated_labels.detach().cpu(),minlength=750)
  self.afa_generated_support+=per_class
  detail.update(generated_count=limit,generated_per_class=per_class.tolist(),different_sample_donors=True,extra_visual_forwards=0)
  return loss,detail
 def method_state_dict(self):
  return dict(schema='aictrack1.afa_CLIP_anchored.training_state.v1',recipe=copy.deepcopy(self.afa_recipe),epoch_index=self.afa_epoch_index,frequency_counts=None if self.afa_class_counts is None else self.afa_class_counts.clone(),generator_state=self.afa_generator.get_state().clone(),observed_support=self.afa_observed_support.clone(),eligible_support=self.afa_eligible_support.clone(),generated_support=self.afa_generated_support.clone(),generation_disabled_at_inference=True)
 def load_method_state_dict(self,state):
  if state.get('schema')!='aictrack1.afa_CLIP_anchored.training_state.v1' or state.get('recipe')!=self.afa_recipe or state.get('generation_disabled_at_inference') is not True:raise ValueError('AFA exact method checkpoint')
  self.set_afa_training_context(state['epoch_index'],state['frequency_counts'])
  rng=state['generator_state']
  if not isinstance(rng,torch.Tensor) or rng.dtype!=torch.uint8 or rng.ndim!=1 or len(rng)==0:raise ValueError('AFA independent generator RNG')
  self.afa_generator.set_state(rng)
  for field,target in [('observed_support','afa_observed_support'),('eligible_support','afa_eligible_support'),('generated_support','afa_generated_support')]:
   v=state[field]
   if not isinstance(v,torch.Tensor) or v.shape!=(750,) or v.dtype!=torch.long or bool((v<0).any()):raise ValueError('AFA actual support ledger')
   setattr(self,target,v.cpu().clone())

def configure_afa(visual,head,*,normalize_features,recipe,device):
 return AFACompatibleClassifier(visual,head,normalize_features=normalize_features,recipe=recipe).to(device)

def afa_parameter_groups(model):
 all_=[p for name,p in model.afa.named_parameters() if name!='gain']
 return [dict(params=all_,lr=1e-4,weight_decay=.01),dict(params=[model.afa.gain],lr=1e-4,weight_decay=0.)]

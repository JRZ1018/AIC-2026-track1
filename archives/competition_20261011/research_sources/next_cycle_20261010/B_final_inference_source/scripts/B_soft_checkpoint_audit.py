from pathlib import Path
import hashlib,math
import torch
ROOT=Path(__file__).resolve().parents[1]
MATH_SOURCE_SHA='0096f4c8352e28eae4373485c45e4e6e4f54dfa750ae4b4ddeca8bd8a068df18'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate_soft_state(state,experiment,train_counts):
 """Validate complete B teacher/state provenance on CPU; export and forward raw student only."""
 from aictrack1.methods.josrc_soft import validate_recipe
 recipe=validate_recipe(experiment['josrc_soft']);arm=state['fresh_route_arm']
 if arm not in ('B25','B50') or recipe['soft_coefficient']!=(.25 if arm=='B25' else .5) or state.get('josrc_math_protocol_sha256')!=sha(ROOT/'inputs/soft_math_protocol.json'):raise ValueError('actual registered B mathematics/coefficient SHA')
 teacher=state.get('josrc_teacher',{});success=teacher.get('successful_updates');skipped=teacher.get('skipped_updates');ema=teacher.get('ema',{})
 if teacher.get('schema')!=recipe['protocol'] or teacher.get('recipe')!=recipe or teacher.get('final_model')!='student_only' or not isinstance(success,int) or success<=0 or not isinstance(skipped,int) or skipped<0 or ema.get('decay')!=.999 or ema.get('updates')!=success:raise ValueError('complete successful-step-only EMA recipe/count')
 raw=state['trainable_state'];shadow=ema.get('shadow',{})
 if set(shadow)!=set(raw):raise ValueError('complete teacher raw student tensor names')
 for name,value in raw.items():
  other=shadow[name]
  if not isinstance(other,torch.Tensor) or other.dtype!=torch.float32 or other.shape!=value.shape or not bool(torch.isfinite(other).all()):raise ValueError('complete finite teacher master '+name)
 history=state['history'];previous_success=0;previous_skipped=0;counts=torch.as_tensor(train_counts,dtype=torch.int64)
 if counts.shape!=(750,) or bool((counts<=0).any()):raise ValueError('full official train750 supports')
 training_count=int(counts.sum());expected_batches=math.ceil(training_count/64)
 for epoch,row in enumerate(history,1):
  detail=row.get('josrc_soft',{});active=epoch>5;ds=detail.get('EMA_successful_updates');dk=detail.get('EMA_skipped_updates');gradient=row.get('gradient_diagnostics',{})
  if detail.get('recipe')!=recipe or detail.get('final_model')!='student_only' or detail.get('soft_active')!=active or detail.get('additional_no_grad_forwards_per_batch')!=(2 if active else 0) or detail.get('hard_branch_dropped_samples')!=0 or detail.get('soft_only_additive') is not True:raise ValueError('all20 original hard supports plus registered auxiliary only')
  if not isinstance(ds,int) or not isinstance(dk,int) or ds<previous_success or dk<previous_skipped or ds-previous_success!=gradient.get('amp_successful_optimizer_updates') or dk-previous_skipped!=gradient.get('amp_skipped_optimizer_updates') or ds-previous_success+dk-previous_skipped!=expected_batches:raise ValueError('teacher success count matches actual AMP every epoch')
  if detail.get('hard_seen')!=(counts*epoch).tolist() or detail.get('teacher_transitions_total')!=max(0,epoch-5)*training_count or detail.get('active_teacher_observations')!=max(0,epoch-5)*training_count:raise ValueError('actual hard and active-only teacher supports every epoch')
  mass=torch.as_tensor(detail.get('soft_mass'),dtype=torch.float64)
  if mass.shape!=(750,) or not bool(torch.isfinite(mass).all()) or bool((mass<0).any()) or bool((mass>counts.double()*max(0,epoch-5)+1e-5).any()):raise ValueError('finite additive soft class mass')
  previous_success,previous_skipped=ds,dk
 if previous_success!=success or previous_skipped!=skipped:raise ValueError('terminal teacher cumulative actual updates')
 moments=state['optimizer']['state']
 for entry in moments.values():
  step=entry.get('step');step=float(step) if isinstance(step,torch.Tensor) else step
  if step!=success:raise ValueError('every fullvisual/head AdamW moment step equals actual teacher successful updates')
 support=state.get('josrc_support',{});hard=support.get('hard_seen');mass=support.get('soft_mass');trans=support.get('teacher_transitions')
 if support.get('num_classes')!=750 or support.get('hard_branch_dropped_samples')!=0 or support.get('soft_only_additive') is not True or not isinstance(hard,torch.Tensor) or hard.dtype!=torch.int64 or not torch.equal(hard,counts*20):raise ValueError('complete terminal20 hard supports')
 if not isinstance(mass,torch.Tensor) or mass.dtype!=torch.float64 or mass.shape!=(750,) or not bool(torch.isfinite(mass).all()) or bool((mass<0).any()) or not torch.equal(mass,torch.tensor(history[-1]['josrc_soft']['soft_mass'],dtype=torch.float64)):raise ValueError('exact last soft mass summary binding')
 if not isinstance(trans,torch.Tensor) or trans.dtype!=torch.int64 or trans.shape!=(750,750) or bool((trans<0).any()) or not torch.equal(trans.sum(1),counts*15) or support.get('active_teacher_observations')!=training_count*15:raise ValueError('real15 active-epoch teacher migrations; no warmup fake predictions')
 return dict(josrc_soft_state_complete=True,raw_student_only=True,teacher_used_for_inference=False,teacher_complete_tensors=len(shadow),EMA_decay=.999,EMA_successful_updates=success,EMA_skipped_updates=skipped,hard_support_all20_checked=True,teacher_active_support_only15_checked=True,math_source_sha256=MATH_SOURCE_SHA,math_protocol_sha256=state['josrc_math_protocol_sha256'],auxiliary_soft_coefficient=recipe['soft_coefficient'],warmup_epochs=5)

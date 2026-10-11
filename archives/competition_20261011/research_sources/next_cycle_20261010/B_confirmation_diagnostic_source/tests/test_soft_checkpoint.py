"""B teacher/counter/stat/method identity corruptions; CPU only and no official model."""
import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT.parent/'confirmation_visual_B_source';sys.path[:0]=[str(ROOT),str(TRAIN/'src'),str(TRAIN/'scripts')]
import torch
import fresh_diag_contract as c
from aictrack1.methods.josrc_soft import recipe

def fixture(arm='B25'):
 counts=torch.full((750,),158,dtype=torch.int64);counts[:458]+=1;total=int(counts.sum());batches=(total+63)//64;spec=recipe({'J5':0.,'B25':.25,'B50':.5}[arm]);epochs=0 if arm=='J5' else 15
 raw={'visual.a':torch.zeros(2),'visual.b':torch.zeros(3),'classifier.linear.weight':torch.zeros(8,5),'classifier.linear.bias':torch.zeros(8)}
 history=[]
 for epoch in range(1,21):
  success=epoch*batches-2;skips=2;active=arm!='J5' and epoch>5;mass=counts.double()*(max(0,epoch-5) if arm!='J5' else 0)*.01
  detail=dict(recipe=spec,final_model='student_only',soft_active=active,additional_no_grad_forwards_per_batch=2 if active else 0,hard_branch_dropped_samples=0,soft_only_additive=True,EMA_successful_updates=success,EMA_skipped_updates=skips,hard_seen=(counts*epoch).tolist(),soft_mass=mass.tolist(),teacher_transitions_total=(max(0,epoch-5) if arm!='J5' else 0)*total,active_teacher_observations=(max(0,epoch-5) if arm!='J5' else 0)*total)
  history.append(dict(epoch=epoch,josrc_soft=detail,gradient_diagnostics=dict(amp_successful_optimizer_updates=batches-(2 if epoch==1 else 0),amp_skipped_optimizer_updates=2 if epoch==1 else 0)))
 transitions=torch.diag(counts*epochs);state=dict(fresh_route_arm=arm,trainable_state=raw,history=history,josrc_math_protocol_sha256=c.sha(ROOT/'inputs/soft_math_protocol.json'),josrc_teacher=dict(schema=spec['protocol'],recipe=spec,final_model='student_only',successful_updates=20*batches-2,skipped_updates=2,ema=dict(decay=.999,updates=20*batches-2,shadow={k:v.clone() for k,v in raw.items()})),josrc_support=dict(num_classes=750,hard_branch_dropped_samples=0,soft_only_additive=True,hard_seen=counts*20,soft_mass=counts.double()*epochs*.01,teacher_transitions=transitions,active_teacher_observations=total*epochs),optimizer=dict(state={i:dict(step=torch.tensor(float(20*batches-2))) for i in range(len(raw))}))
 return state,dict(josrc_soft=spec),counts
class SoftCheckpoint(unittest.TestCase):
 def test_actual_new_controller_no_run_flag(self):
  from phase_guard import actual_command_arms,normalize_owner
  root=c.TRAIN_ROOT;python='/root/autodl-tmp/venvs/aictrack1/bin/python'
  for command in [[python,'-B','scripts/run_confirmation_visual.py'],[python,'scripts/run_confirmation_visual.py'],[python,'-B','scripts/run_confirmation_visual.py','--resume-arms','B25']]:self.assertEqual(actual_command_arms(command,root,'B25'),['J5','B25'])
  for flags in [['--run'],['--smoke','B25'],['--preflight'],['--resume-arms','B50'],['--resume-arms','B25','B25']]:
   with self.assertRaises(ValueError):actual_command_arms([python,'-B','scripts/run_confirmation_visual.py']+flags,root,'B25')
  self.assertEqual(normalize_owner(dict(pid=19,pgrp=19,session=19,start_ticks='24'))['pid'],19)
  for value in [dict(pid='19',pgrp='19',session='19',start_ticks='24'),dict(pid=True,pgrp=True,session=True,start_ticks='24'),dict(pid=19,pgrp=19,session=19,start_ticks=24)]:
   with self.assertRaises(ValueError):normalize_owner(value)
 def test_control_shadow_still_updated_but_zero_auxiliary(self):
  v,e,n=fixture('J5');a=c.validate_soft_state(v,e,n);self.assertEqual(a['teacher_active_epochs_checked'],0);self.assertGreater(a['EMA_successful_updates'],0);self.assertEqual(a['auxiliary_soft_coefficient'],0.)
  v['history'][5]['josrc_soft']['soft_mass'][0]=1e-9
  with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_control_extra_forward_or_teacher_migration_rejected(self):
  for mode in ['forward','migration']:
   v,e,n=fixture('J5')
   if mode=='forward':v['history'][5]['josrc_soft']['additional_no_grad_forwards_per_batch']=2
   else:v['josrc_support']['teacher_transitions'][0,0]=1
   with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_complete_20_CPU_student_only_teacher_stats(self):
  for arm in ['B25','B50']:
   v,e,n=fixture(arm);audit=c.validate_soft_state(v,e,n);self.assertTrue(audit['raw_student_only']);self.assertFalse(audit['teacher_used_for_inference']);self.assertEqual(audit['EMA_successful_updates'],37178)
 def test_method_missing_or_different_rejected(self):
  for action in ['missing','different']:
   v,e,n=fixture();v.pop('josrc_math_protocol_sha256') if action=='missing' else v.update(josrc_math_protocol_sha256='0'*64)
   with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_teacher_omitted_head_wrong_shape_dtype_nan(self):
  for action in ['missing','shape','dtype','nan']:
   v,e,n=fixture();shadow=v['josrc_teacher']['ema']['shadow']
   if action=='missing':shadow.pop('classifier.linear.bias')
   elif action=='shape':shadow['classifier.linear.bias']=torch.zeros(3)
   elif action=='dtype':shadow['visual.a']=shadow['visual.a'].half()
   else:shadow['visual.a'][0]=float('nan')
   with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_skipped_steps_do_not_advance_EMA(self):
  v,e,n=fixture();v['josrc_teacher']['ema']['updates']+=2
  with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_warmup_fake_prediction_migration_rejected(self):
  v,e,n=fixture();v['history'][0]['josrc_soft']['teacher_transitions_total']=118958
  with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_hard_support_loss_or_mask_rejected(self):
  for action in ['drop','seen']:
   v,e,n=fixture()
   if action=='drop':v['josrc_support']['hard_branch_dropped_samples']=1
   else:v['josrc_support']['hard_seen'][0]-=1
   with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_actualAMP_optimizer_and_teacher_counts_bind(self):
  for action in ['row','optimizer']:
   v,e,n=fixture()
   if action=='row':v['history'][8]['gradient_diagnostics']['amp_successful_optimizer_updates']-=1
   else:v['optimizer']['state'][0]['step']-=1
   with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_migrations_bound_to_actual_original_classes(self):
  v,e,n=fixture();v['josrc_support']['teacher_transitions'][0,0]-=1
  with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
 def test_terminal_soft_mass_vs_metrics_history(self):
  v,e,n=fixture();v['josrc_support']['soft_mass'][0]+=.1
  with self.assertRaises(ValueError):c.validate_soft_state(v,e,n)
if __name__=='__main__':unittest.main(verbosity=2)

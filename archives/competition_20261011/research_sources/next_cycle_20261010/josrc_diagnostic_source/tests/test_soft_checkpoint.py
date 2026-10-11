"""B teacher/counter/stat/method identity corruptions; CPU only and no official model."""
import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT.parent/'josrc_soft_r2_source';sys.path[:0]=[str(ROOT),str(TRAIN/'src'),str(TRAIN/'scripts')]
import torch
import fresh_diag_contract as c
from aictrack1.methods.josrc_soft import recipe

def fixture(arm='B25'):
 counts=torch.full((750,),158,dtype=torch.int64);counts[:458]+=1;total=int(counts.sum());batches=(total+63)//64;spec=recipe(.25 if arm=='B25' else .5)
 raw={'visual.a':torch.zeros(2),'visual.b':torch.zeros(3),'classifier.linear.weight':torch.zeros(8,5),'classifier.linear.bias':torch.zeros(8)}
 history=[]
 for epoch in range(1,21):
  success=epoch*batches-2;skips=2;mass=counts.double()*max(0,epoch-5)*.01
  detail=dict(recipe=spec,final_model='student_only',soft_active=epoch>5,additional_no_grad_forwards_per_batch=2 if epoch>5 else 0,hard_branch_dropped_samples=0,soft_only_additive=True,EMA_successful_updates=success,EMA_skipped_updates=skips,hard_seen=(counts*epoch).tolist(),soft_mass=mass.tolist(),teacher_transitions_total=max(0,epoch-5)*total,active_teacher_observations=max(0,epoch-5)*total)
  history.append(dict(epoch=epoch,josrc_soft=detail,gradient_diagnostics=dict(amp_successful_optimizer_updates=batches-(2 if epoch==1 else 0),amp_skipped_optimizer_updates=2 if epoch==1 else 0)))
 transitions=torch.diag(counts*15);state=dict(fresh_route_arm=arm,trainable_state=raw,history=history,josrc_math_protocol_sha256=c.sha(ROOT/'inputs/soft_math_protocol.json'),josrc_teacher=dict(schema=spec['protocol'],recipe=spec,final_model='student_only',successful_updates=20*batches-2,skipped_updates=2,ema=dict(decay=.999,updates=20*batches-2,shadow={k:v.clone() for k,v in raw.items()})),josrc_support=dict(num_classes=750,hard_branch_dropped_samples=0,soft_only_additive=True,hard_seen=counts*20,soft_mass=counts.double()*15*.01,teacher_transitions=transitions,active_teacher_observations=total*15),optimizer=dict(state={i:dict(step=torch.tensor(float(20*batches-2))) for i in range(len(raw))}))
 return state,dict(josrc_soft=spec),counts
class SoftCheckpoint(unittest.TestCase):
 def test_actual_no_schema_root_launch_with_B_Python_flag(self):
  from phase_guard import actual_command_arms
  root=Path('/root/autodl-tmp/AICtrack1_next_cycle_josrc_soft_20261010');python='/root/autodl-tmp/venvs/aictrack1/bin/python'
  actual=dict(mode='train',process=dict(pid=20210,pgrp=20210,session=20210,start_ticks=313600000,cwd=str(root)),bound_source_sha256='f5a402ede5f2e83c2948e358e56eeb5359fec8699712aa7ec2caa880a345c7f9',command=[python,'-B','scripts/run_josrc.py','--run','--arms','B25','B50'],test_access=False)
  self.assertEqual(actual_command_arms(actual['command'],root,'B25'),['B25','B50']);self.assertEqual(actual_command_arms([python,'scripts/run_josrc.py','--run','--arms','B25'],root,'B25'),['B25'])
  for command in [[python,'-c','print(1)','--run','--arms','B25'],[python,'-B','scripts/run_josrc.py','--profile100','--arms','B25'],[python,'-B','scripts/run_josrc.py','--run','--arms','B25','B25']]:
   with self.assertRaises(ValueError):actual_command_arms(command,root,'B25')
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

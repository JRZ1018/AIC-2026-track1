"""Minimal PF visual science/config CPU fixtures; no actual model or images."""
import ast,copy,hashlib,json,os,random,sys,tempfile,unittest
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np,torch
import p1_contract as c
from train_n1_peft import _base_loss_per_sample,_weighted_base_loss
from aictrack1.methods.persistent_filter import PersistentTrainingGate,ActualSupervisionLedger
from PF_admission_contract import validate_confirmation_budget
from bind_confirmation_visual import validate_budget
from pf_fixture import budget,make_admission

class VisualContracts(unittest.TestCase):
 def assets(self):
  return dict(arms=['J5','PF30'],assets={name:dict(path='/actual/same/'+name) for name in ('manifest','split','label_conflicts','h1')})
 def test_truthful_unbound_future_refs(self):
  with self.assertRaises(ValueError):c.verify_files()
  self.assertEqual(c.verify_files(require_bound=False)['status'],'unbound_template');d=c.read_json(ROOT/'inputs/future_binding.json');self.assertIsNone(d['budget_sha256']);self.assertIsNone(d['new_H1_checkpoint_sha256']);self.assertEqual(d['visual_checkpoint_sha256'],{'J5':None,'PF30':None})
 def test_only_filter_switch_scientific_difference(self):
  p=self.assets();control=c.expected_config('J5',p);candidate=c.expected_config('PF30',p)
  self.assertEqual(control['output_dir'],'../../artifacts/experiments/CONFIRM_J5');self.assertEqual(candidate['output_dir'],'../../artifacts/experiments/CONFIRM_PF30');self.assertEqual(control['run_id'],'20261010-confirmation-J5-fixed20');self.assertEqual(candidate['run_id'],'20261010-confirmation-PF30-fixed20')
  self.assertNotIn('persistent_filter',control);self.assertEqual(candidate.pop('persistent_filter'),dict(enabled=True,protocol='persistent-supervision-gate-v2',max_ac_weight=.30))
  for raw in (control,candidate):raw.pop('run_id');raw.pop('output_dir');self.assertEqual(raw['training']['seed'],20261010);self.assertEqual(raw['base_loss'],{'kind':'sce','alpha':.1,'beta':1.,'log_zero':-4.});self.assertEqual(raw['training']['head_learning_rate'],5e-6)
  self.assertEqual(control,candidate)
 def test_wrong_config_or_unregistered_threshold_arm_rejected(self):
  p=self.assets();raw=c.expected_config('PF30',p);raw['persistent_filter']['max_ac_weight']=.5
  with self.assertRaises(ValueError):c.validate_config(raw,'PF30',p)
  raw=c.expected_config('J5',p);raw['training']['seed']=20261008
  with self.assertRaises(ValueError):c.validate_config(raw,'J5',p)
  with self.assertRaises(ValueError):c.expected_config('PF50',p)
 def test_disabled_actual_branch_exact_loss_gradient_RNG_IDs(self):
  text=(ROOT/'scripts/train_n1_peft.py').read_text(encoding='utf-8');tree=ast.parse(text);branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='gate is None' and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in x.targets) for x in n.body))
  logits=torch.tensor([[1.,-.4,.2],[.1,.6,-.2]],requires_grad=True);other=logits.detach().clone().requires_grad_(True);labels=torch.tensor([0,2]);weights=torch.tensor([.05,.8]);ids=torch.tensor([3,9]);ids_before=ids.clone()
  def loss(x):return _base_loss_per_sample(x,labels,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.1,sce_beta=1.,sce_log_zero=-4.)
  expected=_weighted_base_loss(loss(logits),weights);py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state().clone()
  namespace=dict(gate=None,per_sample_loss=loss(other),batch_weights=weights,labels=labels,supervised_samples=0,_weighted_base_loss=_weighted_base_loss)
  exec(compile(ast.Module(body=[branch],type_ignores=[]),'<actual_frozen_disabled_branch>','exec'),namespace)
  self.assertTrue(torch.equal(expected,namespace['loss']));self.assertTrue(torch.equal(torch.autograd.grad(expected,logits)[0],torch.autograd.grad(namespace['loss'],other)[0]));self.assertEqual(namespace['supervised_samples'],2);self.assertTrue(torch.equal(ids,ids_before));self.assertEqual(random.getstate(),py);self.assertTrue(np.array_equal(np.random.get_state()[1],nr[1]));self.assertTrue(torch.equal(torch.get_rng_state(),tr))
 def gate(self):
  ids=torch.arange(100);labels=ids//50;gate=PersistentTrainingGate(ids,labels,max_ac_weight=.30);weights=torch.ones(100);weights[:8]=.2;weights[50:58]=.2
  return gate,ids,labels,weights
 def test_PF_actual_snapshot_distinct_from_following_next(self):
  gate,ids,labels,weights=self.gate();initial_rng=torch.get_rng_state().clone()
  for _ in range(2):gate.update(ids,weights,torch.zeros(100),torch.ones(100))
  ledger=ActualSupervisionLedger(gate,weights);ledger.observe(ids,labels);before=ledger.finish();nxt=gate.update(ids,weights,torch.zeros(100),torch.ones(100));after=ledger.finish()
  self.assertEqual(before,after);self.assertEqual(before['excluded'],0);self.assertGreater(nxt['excluded'],0);self.assertEqual(before['mask_source_completed_epoch'],2);self.assertTrue(torch.equal(torch.get_rng_state(),initial_rng))
 def test_PF_actual_disk_restore_exact_AC_ids_and_next_update(self):
  gate,ids,labels,weights=self.gate()
  for _ in range(3):gate.update(ids,weights,torch.zeros(100),torch.ones(100))
  with tempfile.TemporaryDirectory(prefix='PF_visual_state_') as td:
   path=Path(td)/'SYNTHETIC_gate.pt';torch.save(dict(persistent_filter_state=gate.state_dict(),AC_weights=weights.clone()),path);loaded=torch.load(path,map_location='cpu',weights_only=True);new=PersistentTrainingGate(ids,labels,max_ac_weight=.30);new.load_state_dict(loaded['persistent_filter_state'],ac_weights=loaded['AC_weights'])
   self.assertTrue(torch.equal(new.keep,gate.keep));self.assertTrue(torch.equal(new.streak,gate.streak));self.assertEqual(new.update(ids,weights,torch.ones(100),torch.ones(100)),gate.update(ids,weights,torch.ones(100),torch.ones(100)))
   bad=copy.deepcopy(loaded);bad['AC_weights'][0]=.25
   with self.assertRaises(ValueError):new.load_state_dict(bad['persistent_filter_state'],ac_weights=bad['AC_weights'])
   with self.assertRaises(ValueError):new.load_state_dict(loaded['persistent_filter_state'],ac_weights=weights[torch.arange(99,-1,-1)])
 def test_actual_common_shared_budget(self):
  with tempfile.TemporaryDirectory(prefix='PF_visual_budget_') as td:
   original=budget(td);self.assertEqual(validate_budget(original),validate_confirmation_budget(original))
   changed=copy.deepcopy(original);changed['final_gpu_reserve_hours']=3
   with self.assertRaises(ValueError):validate_budget(changed)
 def test_actual_PF_science_and_data_layout_bytes(self):
  m=c.read_json(ROOT/'inputs/PF_exploration_source_manifest.json')
  for rel in ['scripts/train_n1_peft.py','src/aictrack1/methods/persistent_filter.py','scripts/train_peft.py','scripts/owned_launch.py','src/aictrack1/engine/training_queue.py','src/aictrack1/data/indexed_images.py','src/aictrack1/methods/confidence.py','src/aictrack1/methods/sce.py']:
   self.assertEqual(c.sha(ROOT/rel),m['files'][rel],rel)
  parent=c.read_json(ROOT/'inputs/parent_generic_template_manifest.json')
  for rel in ['scripts/mainline_h1_contract.py','scripts/p1_smoke.py','scripts/launch_gate.py','scripts/mainline_guard.py','inputs/confirmation_rows.csv','inputs/confirmation_split.csv','inputs/query_identity.json']:
   self.assertEqual(c.sha(ROOT/rel),parent['files'][rel],rel)
if __name__=='__main__':unittest.main(verbosity=2)

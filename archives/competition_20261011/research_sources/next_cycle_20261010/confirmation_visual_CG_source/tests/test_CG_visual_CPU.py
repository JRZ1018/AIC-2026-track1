"""Eight minimal CG confirmation CPU checks; synthetic metadata/tiny state only, no model/image/server/GPU."""
import ast,copy,io,json,os,random,sys,tempfile,unittest
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np,torch,yaml
import p1_contract as c
from aictrack1.data.contracts import load_dataset_config
from aictrack1.data.transforms import center_guard_identity,validate_center_guard_resume
from train_n1_peft import _base_loss_per_sample,_weighted_base_loss
from CG_admission_contract import validate_confirmation_budget
from bind_confirmation_visual import validate_budget
from cg_fixture import budget,make_admission
class VisualContracts(unittest.TestCase):
 def assets(self):return dict(arms=['J5','CG25'],assets={name:dict(path='/actual/same/'+name) for name in ('manifest','split','label_conflicts','h1')})
 def test_truthful_unbound_future_refs(self):
  with self.assertRaises(ValueError):c.verify_files()
  m=c.verify_files(require_bound=False);self.assertEqual(m['status'],'unbound_template');self.assertEqual(m['future_checkpoint_sha256'],{'J5':None,'CG25':None});d=c.read_json(ROOT/'inputs/future_binding.json');self.assertIsNone(d['budget_sha256']);self.assertIsNone(d['new_H1_checkpoint_sha256']);self.assertEqual(d['visual_checkpoint_sha256'],{'J5':None,'CG25':None})
 def test_only_center_guard_one_variable_and_same_common_seed(self):
  p=self.assets();control=c.expected_config('J5',p);candidate=c.expected_config('CG25',p)
  for arm,raw in [('J5',control),('CG25',candidate)]:
   self.assertEqual(raw['dataset_config'],f'../data/p1_{arm}_336.yaml');self.assertEqual(raw['training']['seed'],20261010);self.assertEqual(raw['training']['epochs'],20);self.assertEqual(raw['base_loss'],{'kind':'sce','alpha':.1,'beta':1.,'log_zero':-4.});self.assertNotIn('persistent_filter',raw);self.assertNotIn('josrc_soft',raw)
   raw.pop('dataset_config');raw.pop('run_id');raw.pop('output_dir')
  self.assertEqual(control,candidate)
  a=yaml.safe_load((ROOT/'configs/data/p1_J5_336.yaml').read_text());b=yaml.safe_load((ROOT/'configs/data/p1_CG25_336.yaml').read_text());self.assertEqual(a['dataset']['transforms'].pop('center_guard_fraction'),0.);self.assertEqual(b['dataset']['transforms'].pop('center_guard_fraction'),.25);self.assertEqual(a,b)
 def test_wrong_config_seed_and_unregistered_arm(self):
  p=self.assets();raw=c.expected_config('CG25',p);raw['dataset_config']='../data/p1_J5_336.yaml'
  with self.assertRaises(ValueError):c.validate_config(raw,'CG25',p)
  raw=c.expected_config('J5',p);raw['training']['seed']=20261008
  with self.assertRaises(ValueError):c.validate_config(raw,'J5',p)
  with self.assertRaises(ValueError):c.expected_config('PF30',p)
 def test_exact_CG_science_and_original_GO_query_bytes(self):
  m=c.read_json(ROOT/'inputs/CG_exploration_template_source_manifest.json')
  for rel in ['scripts/train_n1_peft.py','src/aictrack1/data/contracts.py','src/aictrack1/data/transforms.py','scripts/train_peft.py','scripts/owned_launch.py','src/aictrack1/engine/training_queue.py','src/aictrack1/data/indexed_images.py','src/aictrack1/methods/confidence.py','src/aictrack1/methods/sce.py']:
   self.assertEqual(c.sha(ROOT/rel),m['files'][rel],rel)
  parent=c.read_json(ROOT/'inputs/parent_generic_template_manifest.json')
  for rel in ['scripts/mainline_h1_contract.py','scripts/p1_smoke.py','scripts/launch_gate.py','scripts/mainline_guard.py','inputs/confirmation_rows.csv','inputs/confirmation_split.csv','inputs/query_identity.json']:
   self.assertEqual(c.sha(ROOT/rel),parent['files'][rel],rel)
 def test_actual_augmentation_state_BytesIO_and_wrong_restore(self):
  d=load_dataset_config(ROOT/'configs/data/p1_CG25_336.yaml');identity=center_guard_identity(d.transforms,dataset_sha256=c.sha(ROOT/'configs/data/p1_CG25_336.yaml'),source_sha256=c.sha(ROOT/'src/aictrack1/data/transforms.py'));state={'center_guard_augmentation':identity};f=io.BytesIO();torch.save(state,f);f.seek(0);loaded=torch.load(f,map_location='cpu',weights_only=True);validate_center_guard_resume(loaded,identity);self.assertEqual(identity['center_guard_fraction'],.25)
  for field in ('center_guard_fraction','dataset_config_sha256','transform_source_sha256','stochastic_seed_protocol','random_draws'):
   bad=copy.deepcopy(loaded);bad['center_guard_augmentation'][field]='mismatch'
   with self.assertRaises(ValueError):validate_center_guard_resume(bad,identity)
  control=load_dataset_config(ROOT/'configs/data/p1_J5_336.yaml');self.assertEqual(control.transforms.center_guard_fraction,0.)
 def test_original_loss_math_no_extra_branch_or_rng(self):
  raw=c.expected_config('CG25',self.assets());self.assertNotIn('persistent_filter',raw);self.assertFalse(raw.get('trainable_ema',{}).get('enabled',False))
  text=(ROOT/'scripts/train_n1_peft.py').read_text(encoding='utf-8');tree=ast.parse(text);branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='gate is None' and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in x.targets) for x in n.body))
  logits=torch.tensor([[1.,-.4,.2],[.1,.6,-.2]],requires_grad=True);other=logits.detach().clone().requires_grad_(True);labels=torch.tensor([0,2]);weights=torch.tensor([.05,.8]);ids=torch.tensor([3,9]);py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state().clone()
  def per_sample(x):return _base_loss_per_sample(x,labels,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.1,sce_beta=1.,sce_log_zero=-4.)
  expected=_weighted_base_loss(per_sample(logits),weights);namespace=dict(gate=None,per_sample_loss=per_sample(other),batch_weights=weights,labels=labels,supervised_samples=0,_weighted_base_loss=_weighted_base_loss)
  exec(compile(ast.Module(body=[branch],type_ignores=[]),'<actual_CG_unchanged_disabled_legacy_branch>','exec'),namespace)
  self.assertTrue(torch.equal(expected,namespace['loss']));self.assertTrue(torch.equal(torch.autograd.grad(expected,logits)[0],torch.autograd.grad(namespace['loss'],other)[0]));self.assertEqual(namespace['supervised_samples'],2);self.assertTrue(torch.equal(ids,torch.tensor([3,9])));self.assertEqual(random.getstate(),py);self.assertTrue(np.array_equal(np.random.get_state()[1],nr[1]));self.assertTrue(torch.equal(torch.get_rng_state(),tr))
 def test_same_future_budget_no_reset(self):
  with tempfile.TemporaryDirectory(prefix='CG_visual_budget_') as td:
   original=budget(td);self.assertEqual(validate_budget(original),validate_confirmation_budget(original));changed=dict(original,final_gpu_reserve_hours=3)
   with self.assertRaises(ValueError):validate_budget(changed)
 def test_no_PF_B_alpha_gate_borrowing(self):
  with tempfile.TemporaryDirectory(prefix='CG_visual_admission_') as td:
   value,_,_=make_admission(td);self.assertEqual(c.admission(value)['selected_arm'],'CG25')
   for schema in ('aictrack1.next_cycle10.PF_admission.v1','aictrack1.next_cycle10.B_admission.v1','aictrack1.route_admission.v1'):
    with self.assertRaises(ValueError):c.admission(dict(value,schema=schema))
if __name__=='__main__':unittest.main(verbosity=2)

"""CPU-only contract/math/profile restore fixtures; no real images/model/CUDA."""
import ast,copy,hashlib,json,os,random,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import numpy as np
import torch
from torch import nn
from aictrack1.methods.josrc_soft import recipe,SoftTeacher,SupportLedger,combine_loss,js_divergence
from aictrack1.methods.sce import symmetric_cross_entropy
import p1_contract as contract
import profile_josrc100 as profile
class Fixture(unittest.TestCase):
 def setUp(self):random.seed(54);np.random.seed(54);torch.manual_seed(54);torch.set_num_threads(4)
 def test_coefficient_zero_loss_grad_exact(self):
  a=torch.randn(8,8,requires_grad=True);b=a.detach().clone().requires_grad_();y=torch.arange(8);weights=torch.linspace(.2,1.,8)
  hard=lambda z:(symmetric_cross_entropy(z,y,alpha=.1,beta=1.,log_zero=-4.,reduction='none')*weights).sum()/weights.sum()
  baseline=hard(a);original=hard(b);mixed,_=combine_loss(original,b,None,None,y,recipe(0.),5);self.assertIs(mixed,original);self.assertTrue(torch.equal(baseline,mixed));self.assertTrue(torch.equal(torch.autograd.grad(baseline,a)[0],torch.autograd.grad(mixed,b)[0]))
 def test_original_helper_hard_math_is_identical(self):
  base=ast.parse((ROOT/'inputs/original_train_n1_peft.py').read_text(encoding='utf-8'));new=ast.parse((ROOT/'scripts/train_n1_peft.py').read_text(encoding='utf-8'))
  funcs=['_base_loss_per_sample','_weighted_base_loss','_loader','_data_loader_settings','_runtime_settings']
  for name in funcs:self.assertEqual(ast.dump(next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name==name)),ast.dump(next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)))
 def test_full_original_hard_AC_block_literal_preserved(self):
  original=(ROOT/'inputs/original_train_n1_peft.py').read_text(encoding='utf-8');new=(ROOT/'scripts/train_n1_peft.py').read_text(encoding='utf-8')
  for begin,end in [('                logits = model(images)','            if not torch.isfinite(loss):'),('            if confidence is not None:\n                detached_logits','        scheduler.step()')]:
   old=original[original.index(begin):original.index(end,original.index(begin))]
   if begin.startswith('                logits'):self.assertIn(old,new)
   else:self.assertIn(old,new)
 def test_config_only_auxiliary_diff_and_one_control_branch(self):
  import yaml
  original=yaml.safe_load((ROOT/'inputs/baseline_J5.yaml').read_text(encoding='utf-8'))['experiment']
  for arm,value in [('B25',.25),('B50',.5)]:
   new=contract.expected_config(arm);self.assertEqual(new['josrc_soft'],recipe(value));new.pop('josrc_soft');new['run_id']=original['run_id'];new['output_dir']=original['output_dir'];self.assertEqual(new,original)
   self.assertEqual(new['base_loss'],dict(kind='sce',alpha=.1,beta=1.,log_zero=-4.));self.assertNotIn('trainable_ema',new)
 def test_actual_template_assets_metadata_and_both_config_closure(self):
  value=contract.assets(ROOT);self.assertEqual(value['h1_row_binding']['counts'],dict(train=118958,validation=29737));self.assertFalse(value['h1_row_binding']['validation_used_for_optimization_or_selection'])
  receipt=json.loads((ROOT/'inputs/h1_actual_CPU_complete.json').read_text(encoding='utf-8'));self.assertEqual(value['h1_row_binding'],receipt['producer_receipt']['row_binding'])
  for arm in contract.ARMS:contract.validate_config(contract.expected_config(arm,value),arm,value)
  manifest=json.loads((ROOT/'inputs/previous_r1_source_manifest.json').read_text(encoding='utf-8'));self.assertEqual(manifest['future_budget_sha256'],None);self.assertFalse(value['test_access'])
 def test_JS_onehot_extremes_and_invalid(self):
  p=torch.eye(4);self.assertTrue(torch.equal(js_divergence(p,p),torch.zeros(4)));self.assertTrue(torch.isfinite(js_divergence(p,p.roll(1,0))).all())
  with self.assertRaises(ValueError):js_divergence(p*2,p)
 def test_warmup_updates_teacher_skip_does_not(self):
  model=nn.Sequential(nn.Linear(5,8),nn.Tanh());teacher=SoftTeacher(model,recipe(.25));self.assertEqual(teacher.weak_views_before_strong(torch.ones(8,5),4),(None,None));teacher.update_after_optimizer(True);v=copy.deepcopy(teacher.state_dict());teacher.update_after_optimizer(False);self.assertEqual(teacher.successful_updates,1);self.assertEqual(teacher.skipped_updates,1);self.assertEqual(profile.maximum_difference(v['ema'],teacher.state_dict()['ema']),0)
 def test_profile_actual_state_safe_disk_nextstep(self):
  model=nn.Sequential(nn.Linear(5,7),nn.Tanh(),nn.Dropout(.3),nn.Linear(7,8));model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.002);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,20);scaler=torch.amp.GradScaler('cpu',enabled=False);teacher=SoftTeacher(model,recipe(.25));support=SupportLedger(8);x=torch.randn(8,5);y=torch.arange(8)
  def step():
   optimizer.zero_grad();sw,tw=teacher.weak_views_before_strong(x,5);logits=model(x);hard=symmetric_cross_entropy(logits,y,alpha=.1,beta=1.,log_zero=-4.);loss,d=combine_loss(hard,logits,sw,tw,y,recipe(.25),5);loss.backward();optimizer.step();teacher.update_after_optimizer(True);support.observe(y,d);return loss.detach()
  step();saved=profile.state(model,teacher,support,optimizer,scheduler,scaler)
  with tempfile.TemporaryDirectory() as temp:
   file=Path(temp)/'profile.pt';torch.save(saved,file);loss=step();expected=profile.state(model,teacher,support,optimizer,scheduler,scaler);loaded=torch.load(file,map_location='cpu',weights_only=True);profile.restore(loaded,model,teacher,support,optimizer,scheduler,scaler);replayed=step();actual=profile.state(model,teacher,support,optimizer,scheduler,scaler)
  self.assertEqual(profile.maximum_difference(expected,actual),0);self.assertTrue(torch.equal(loss,replayed));self.assertEqual(teacher.successful_updates,2);self.assertEqual(int(support.hard_seen.sum()),16)
 def test_same_weak_views_do_not_shift_next_strong_or_mode(self):
  model=nn.Sequential(nn.Linear(5,8),nn.Tanh(),nn.Dropout(.5));model.train();teacher=SoftTeacher(model,recipe(.25));x=torch.randn(8,5);rng=profile.rng_state();baseline=model(x);profile.restore_rng(rng);teacher.weak_views_before_strong(x,5);actual=model(x);self.assertTrue(torch.equal(baseline,actual));actual.sum().backward();self.assertTrue(all(p.grad is not None for p in model.parameters()))
 def test_original_strong_synthetic_image_dataset_exact(self):
  from PIL import Image
  from aictrack1.data.contracts import load_dataset_config
  from aictrack1.data.transforms import build_split_transforms
  from aictrack1.data.indexed_images import IndexedImage,IndexedImageDataset
  from aictrack1.methods.confidence_observation import WeakObservationDataset
  transforms=build_split_transforms(load_dataset_config(ROOT/'configs/data/p1_336.yaml').transforms)
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'fixture.jpg';pixels=np.random.default_rng(17).integers(0,255,size=(410,480,3),dtype=np.uint8);Image.fromarray(pixels).save(path)
   records=[IndexedImage(0,'fixture','fixture.jpg',path,'train','train',0,'0000')];base=IndexedImageDataset(records,transforms['train'],stochastic_seed=20261008);base.set_epoch(5);weak=WeakObservationDataset(base,transforms['validation'],20261008)
   rng=profile.rng_state();a=base[0];before=profile.rng_state();b=weak[0];after=profile.rng_state()
  self.assertTrue(torch.equal(a['image'],b['image']));self.assertEqual(a['sample_id'],b['sample_id']);self.assertEqual(a['sample_index'],b['sample_index']);self.assertEqual(profile.maximum_difference(before,after),0);self.assertEqual(tuple(b['confidence_observation_image'].shape),(3,336,336))
 def test_CPU_cli_entry_imports(self):
  for name in ['train_n1_peft.py','bind_josrc_source.py','run_josrc.py']:
   result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/name),'--help'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8');self.assertEqual(result.returncode,0,(name,result.stderr));self.assertIn('usage:',result.stdout)
 def test_source_hard_branch_154_raw_student_final(self):
  s=(ROOT/'scripts/train_n1_peft.py').read_text(encoding='utf-8');self.assertLess(s.index('student_weak_logits,teacher_weak_logits='),s.index('                logits = model(images)'));self.assertIn('if ema_enabled or raw.get("persistent_filter"',s);self.assertIn('josrc_math_protocol_sha256',s);self.assertIn('step_completed=optimizer_step_completed',s)
  profiletext=(ROOT/'scripts/profile_josrc100.py').read_text(encoding='utf-8');self.assertIn("partitions=['train']",profiletext);self.assertIn('candidate=False',profiletext);self.assertIn("actual_warmup_completed=False",profiletext)
if __name__=='__main__':unittest.main(verbosity=2)

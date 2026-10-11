"""CPU AFA method/restoration and original fixed-numeric contracts; no real models/images."""
from pathlib import Path
import copy,io,json,os,random,subprocess,sys,unittest
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT.parent/'afa_compatible_source'
sys.path[:0]=[str(ROOT),str(TRAIN/'scripts'),str(TRAIN/'src')]
import numpy as np,torch
from torch import nn
import fresh_diag_contract as c
c.TRAIN_ROOT=TRAIN
from afa_checkpoint import validate_method,validate_AFA_optimizer,EnhancedCLSFeatureCollector,expected_afa_recipe,validate_restored_method
from compare_graph_vs_J5 import imported_graph_predictions,graph_scores
from paired_metrics import load_queries,compare_arrays
from run_exploration_diagnostic import dependency_probe
from aictrack1.models.afa_compatible import FeatureDecomposition,AFACompatibleClassifier
from aictrack1.models.heads import LinearClassifier
from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(4);_,cls.parts=c.confirmation_rows();cls.counts=np.bincount([int(r['target']) for r in cls.parts['train']],minlength=750)
 def method(self):
  counts=torch.as_tensor(self.counts);history=[];eligible=torch.zeros(750,dtype=torch.long);generated=eligible.clone()
  for epoch in range(1,21):
   if epoch>10:eligible[0]+=1;generated[0]+=2
   history.append(dict(epoch=epoch,validation={},afa_compatible=dict(epoch_generated=2 if epoch>10 else 0,epoch_trust_eligible=1 if epoch>10 else 0,observed_support=(counts*epoch).tolist(),eligible_support=eligible.tolist(),generated_support=generated.tolist(),residual_gain=0.,class_residual_branch=True,anchored_variation_branch=True,extra_visual_forward_calls=0),gradient_diagnostics=dict(amp_successful_optimizer_updates=1858 if epoch==1 else 1859,amp_skipped_optimizer_updates=1 if epoch==1 else 0)))
  return dict(afa_protocol_sha256=c.sha(TRAIN/'inputs/afa_protocol.json'),afa_training_state=dict(schema='aictrack1.afa_CLIP_anchored.training_state.v1',recipe=expected_afa_recipe(TRAIN),epoch_index=19,frequency_counts=counts,generator_state=torch.Generator().manual_seed(4).get_state(),observed_support=counts*20,eligible_support=eligible,generated_support=generated,generation_disabled_at_inference=True),history=history)
 def state(self):
  extra={'afa.'+n:p.detach().clone() for n,p in FeatureDecomposition().named_parameters()}
  return dict({'visual.a':torch.zeros(2),'visual.b':torch.ones(3),'classifier.linear.weight':torch.zeros(750,512),'classifier.linear.bias':torch.zeros(750)},**extra)
 def optimizer(self):
  s=self.method();s['trainable_state']=self.state();ordered=['visual.a','visual.b','classifier.linear.weight','classifier.linear.bias']+[n for n in s['trainable_state'] if n.startswith('afa.') and n!='afa.gain']+['afa.gain'];groups=[];offset=0
  for count,lr,wd in zip([2,2,10,1],[1.75e-6,5e-6,1e-4,1e-4],[.01,.01,.01,0.],strict=True):
   groups.append(dict(params=list(range(offset,offset+count)),betas=(.9,.999),eps=1e-8,weight_decay=wd,initial_lr=lr,lr=0.));offset+=count
  s['optimizer']=dict(param_groups=groups,state={i:dict(step=torch.tensor(18590. if n.startswith('afa.') else 37179.),exp_avg=torch.zeros_like(s['trainable_state'][n]),exp_avg_sq=torch.zeros_like(s['trainable_state'][n])) for i,n in enumerate(ordered)})
  s.update(scheduler=dict(T_max=20,last_epoch=20,base_lrs=[1.75e-6,5e-6,1e-4,1e-4]),scaler=dict(scale=128.),confidence_weighting=dict(epochs_updated=20));return s
 def test_01_template_source_assets_closed(self):
  self.assertIsNone(c.TRAINING_SOURCE_SHA);c.verify_training_binding(TRAIN,c.TRAINING_TEMPLATE_SHA,allow_prepared=True)
  with self.assertRaises(ValueError):c.verify_training_binding(TRAIN,c.TRAINING_TEMPLATE_SHA)
  self.assertEqual(c.verify_owned_sources(),c.sha(ROOT/'source_manifest.json'))
 def test_02_all11_module_complete(self):
  r=c.validate_state(self.state(),arm='AFA',visual_count=2,visual_numel=5);self.assertEqual(r['afa_new_tensors'],11);self.assertEqual(r['afa_new_parameters'],63613)
 def test_03_bad_AFA_tensor_rejected(self):
  for key,value in [('afa.gain',torch.tensor(float('nan'))),('afa.value.bias',torch.zeros(100,dtype=torch.float16)),('afa.query.weight',torch.zeros(3,3))]:
   s=self.state();s[key]=value
   with self.assertRaises(ValueError):c.validate_state(s,arm='AFA',visual_count=2,visual_numel=5)
  s=self.state();del s['afa.gain']
  with self.assertRaises(ValueError):c.validate_state(s,arm='AFA',visual_count=2,visual_numel=5)
 def test_04_epoch19_method_all_train_support_and_steps(self):
  r=validate_method(self.method(),dict(afa_compatible=expected_afa_recipe(TRAIN)),training_root=TRAIN,train_counts=self.counts)
  self.assertEqual(r['afa_successful_original_updates'],37179);self.assertEqual(r['afa_successful_active_updates'],18590);self.assertTrue(r['afa_generation_disabled_at_inference'])
 def test_05_method_recipe_epoch_protocol_rng_reject(self):
  for field,value in [('epoch_index',0),('recipe',{}),('generation_disabled_at_inference',False),('frequency_counts',torch.ones(750,dtype=torch.long)),('generator_state',torch.zeros(2))]:
   s=self.method();s['afa_training_state'][field]=value
   with self.assertRaises((ValueError,RuntimeError)):validate_method(s,dict(afa_compatible=expected_afa_recipe(TRAIN)),training_root=TRAIN,train_counts=self.counts)
  s=self.method();s['afa_protocol_sha256']='a'*64
  with self.assertRaises(ValueError):validate_method(s,dict(afa_compatible=expected_afa_recipe(TRAIN)),training_root=TRAIN,train_counts=self.counts)
 def test_06_bad_support_and_warmup_rejected(self):
  for mode in ['observed','unsupported_generated','warmup','AMP']:
   s=self.method()
   if mode=='observed':s['history'][10]['afa_compatible']['observed_support'][0]-=1
   elif mode=='unsupported_generated':s['history'][10]['afa_compatible']['generated_support'][1]=1
   elif mode=='warmup':s['history'][0]['afa_compatible']['epoch_generated']=1
   else:s['history'][10]['gradient_diagnostics']['amp_successful_optimizer_updates']=1
   with self.assertRaises(ValueError):validate_method(s,dict(afa_compatible=expected_afa_recipe(TRAIN)),training_root=TRAIN,train_counts=self.counts)
 def test_07_four_optimizer_groups_and_active_only_step(self):
  s=self.optimizer();r=validate_AFA_optimizer(s,list(s['trainable_state']),visual_names=['visual.a','visual.b'],training_root=TRAIN,train_counts=self.counts);self.assertEqual(r['optimizer_parameter_groups'],[2,2,10,1]);self.assertEqual(r['afa_module_moment_steps'],18590)
 def test_08_wrong_moment_step_and_hyperparam_reject(self):
  for mode in ['module','visual','decay','baseLR','scheduler','moment']:
   s=self.optimizer()
   if mode=='module':s['optimizer']['state'][14]['step']=torch.tensor(37179.)
   elif mode=='visual':s['optimizer']['state'][0]['step']=torch.tensor(18590.)
   elif mode=='decay':s['optimizer']['param_groups'][3]['weight_decay']=.01
   elif mode=='baseLR':s['optimizer']['param_groups'][2]['initial_lr']=1e-3
   elif mode=='scheduler':s['scheduler']['last_epoch']=19
   else:del s['optimizer']['state'][4]['exp_avg']
   with self.assertRaises(ValueError):validate_AFA_optimizer(s,list(s['trainable_state']),visual_names=['visual.a','visual.b'],training_root=TRAIN,train_counts=self.counts)
 def test_09_private_global_RNG_check(self):
  raw=np.random.RandomState(3).get_state();s=dict(python_rng_state=random.Random(3).getstate(),numpy_rng_state=(raw[0],raw[1].tolist(),*raw[2:]),torch_rng_state=torch.Generator().manual_seed(3).get_state(),cuda_rng_state_all=[torch.ones(32,dtype=torch.uint8)]);before=torch.get_rng_state().clone();c.validate_rng_state(s);self.assertTrue(torch.equal(before,torch.get_rng_state()))
  for key in s:
   b=copy.deepcopy(s);del b[key]
   with self.assertRaises(ValueError):c.validate_rng_state(b)
 def test_10_actual_query_graph_controls_original(self):
  rows=self.parts['validation'];identity=json.loads((ROOT/'inputs/legacy_control/J5/input_identity.json').read_text(encoding='utf-8'));ctrl=dict(targets=np.array([int(r['target']) for r in rows]),groups=[r['group'] for r in rows],ids=[r['sample_id'] for r in rows],scores=np.empty((0,750)),identity=identity)
  q,qi=load_queries(ctrl,ROOT/'inputs/frozen_queries');self.assertEqual(len(q['balanced_indices']),14760);self.assertEqual(len(q['natural_indices']),29737);pred,p=imported_graph_predictions(q);self.assertEqual(pred['balanced_supported_unique'].shape,(14760,));self.assertEqual(qi['selection_protocol_sha256'],c.sha(ROOT/'selection_protocol.json'))
 def test_11_graph_protection_and_imported_control(self):
  scores=np.array([[8.,0.],[.1,0.],[0.,.1]],dtype=np.float32);f=np.array([[1.,0.]]*3,dtype=np.float32);v,d=graph_scores(scores,f);self.assertTrue(np.array_equal(v[0],scores[0]))
  data=dict(scores=np.array([[6.,0.],[0.,6.]],dtype=np.float32),targets=np.array([0,1]),train_counts=np.array([1,1]),ids=['a','b'],groups=['a','b']);q=dict(natural_indices=np.arange(2),natural_columns=np.arange(2),balanced_indices=np.arange(2),balanced_columns=np.arange(2));override={'balanced_supported_unique':np.array([1,0])};row=compare_arrays(data,data,q,draws=0,only_protocol='balanced_supported_unique',control_predictions=override)[0];self.assertTrue(np.array_equal(row['predictions']['control_p7'],override['balanced_supported_unique']))
 def tiny(self):
  class Visual(nn.Module):
   def __init__(self):super().__init__();self.patch=nn.Linear(8,768);self.transformer=nn.Identity();self.ln_post=nn.LayerNorm(768);self.proj=nn.Parameter(torch.randn(768,512)/30);self.calls=0
   def forward(self,x):self.calls+=1;t=self.transformer(self.patch(x).unsqueeze(0).expand(101,-1,-1));return self.ln_post(t[0])@self.proj
  return AFACompatibleClassifier(Visual(),LinearClassifier(512,750),normalize_features=True)
 def test_12_permanent_epoch19_allparams_safe_restore(self):
  model=self.tiny();model.load_method_state_dict(self.method()['afa_training_state']);model.afa.gain.data.fill_(.4);state=trainable_state_dict(model);stream=io.BytesIO();torch.save(dict(model=state,method=model.method_state_dict()),stream);stream.seek(0);saved=torch.load(stream,weights_only=True);other=self.tiny();other.load_method_state_dict(saved['method']);load_trainable_state_dict(other,saved['model']);other.eval()
  for n,p in other.named_parameters():self.assertTrue(torch.equal(p.detach(),state[n]))
  self.assertEqual(other.afa_epoch_index,19);self.assertTrue(torch.equal(other.afa_generator.get_state(),model.afa_generator.get_state()))
  self.assertTrue(validate_restored_method(other,dict(afa_training_state=saved['method']))['afa_method_restored_equal'])
  other.afa_epoch_index=0
  with self.assertRaises(ValueError):validate_restored_method(other,dict(afa_training_state=saved['method']))
 def test_13_enhanced_sharedhead_4V_observe_no_extra_forward(self):
  model=self.tiny().eval();model.load_method_state_dict(self.method()['afa_training_state']);model.afa.gain.data.fill_(.4);collector=EnhancedCLSFeatureCollector(model)
  try:
   images=torch.randn(2,8)
   for _ in range(4):model(images)
   features=collector.finish(2);self.assertEqual(model.visual.calls,4);self.assertEqual(collector.visual_calls,4);self.assertEqual(collector.calls,4);self.assertEqual(features.shape,(2,512));self.assertIsNone(model.afa_cache)
  finally:collector.close()
 def test_14_real_vendor_dependency_parser_imports(self):
  r=dependency_probe(TRAIN,c.TRAINING_TEMPLATE_SHA,allow_prepared=True);self.assertTrue(r['real_vendor_cli_help_passed']);self.assertFalse(r['checkpoint_opened']);self.assertFalse(r['model_loaded'])
 def test_15_new4h_budget_and_exact_queue(self):
  from phase_guard import validate_budget_payload,validate_actual_launch
  future=json.loads((ROOT/'inputs/budget.json').read_text())
  with self.assertRaises(ValueError):validate_budget_payload(future)
  b=dict(schema='aictrack1.AFA_CLIP_anchored.budget.v1',root_authorized=True,phase_start_unix=100.,deadline_unix=14500.,initial_free_bytes=13013950464,capacity_increase_bytes=32212254720,adjusted_initial_free_bytes=45226205184,data_total_bytes=128849018880,disk_max_bytes=32212254720,min_free_bytes=3221225472,atomic_reserve_bytes=2147483648,final_gpu_reserve_hours=4,test_access=False,capacity_evidence=dict(path='/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_continuation_20261009/root_actual_disk_expansion_recovery_receipt.json',sha256='ba819e4b72731e810743deb1b4b4849eb070e8c664187aeaf255ab3840a78e52'));validate_budget_payload(b)
  for change in [{'deadline_unix':14501.},{'phase_start_unix':True},{'disk_max_bytes':1}]:
   with self.assertRaises(ValueError):validate_budget_payload(dict(b,**change))
  source='a'*64;owner=dict(pid=7,pgrp=7,session=7,start_ticks='123',cwd='/root/autodl-tmp/AICtrack1_next_cycle_AFA_CLS_anchored_20261010');v=dict(bound_source_sha256=source,test_access=False,mode='train',command=['python','-B','scripts/run_supervision_joint.py'],process=owner);validate_actual_launch(v,source)
  for change in [{'command':['python','scripts/run_supervision_joint.py','--profile']},{'process':dict(owner,pid='7')},{'planned_arms':['CG25']}]:
   with self.assertRaises(ValueError):validate_actual_launch(dict(v,**change),source)
 def test_16_original_scientific_math_files_unchanged(self):
  p=json.loads((ROOT/'inputs/parent_diagnostic_source_manifest.json').read_text())
  for rel in ['selection_protocol.json','paired_metrics.py','view_features.py','reference/fv2_single_model.py','reference/diagnose_stage2_tta.py','reference/p0_core.py','reference/p3_graph.py','fixed_graph_protocol.json','frozen_postprocess.json']:
   self.assertEqual(c.sha(ROOT/rel),p['files'][rel])
 def test_17_CPU_CLI_entries(self):
  for script in ['run_exploration_diagnostic.py','compare_graph_vs_J5.py','bind_diagnostic.py']:
   r=subprocess.run([sys.executable,'-B',str(ROOT/script),'--help'],capture_output=True,text=True,timeout=60);self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main(verbosity=2)

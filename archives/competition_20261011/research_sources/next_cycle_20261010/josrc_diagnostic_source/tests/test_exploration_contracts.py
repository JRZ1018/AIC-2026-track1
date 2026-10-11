"""Synthetic CPU contracts and real frozen import/parser checks; no real checkpoint/GPU/images."""
from __future__ import annotations
import copy,hashlib,json,os,random,subprocess,sys,tempfile,unittest
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];TRAIN=ROOT.parent/'josrc_soft_r2_source'
sys.path[:0]=[str(ROOT),str(TRAIN/'scripts'),str(TRAIN/'src')]
import numpy as np
import torch
import fresh_diag_contract as c
from compare_graph_vs_J5 import imported_graph_predictions,graph_scores
from paired_metrics import load_queries,compare_arrays,frozen_decode
from run_exploration_diagnostic import dependency_probe,engine_argv
from view_features import FourViewFeatureCollector,FEATURE_PROTOCOL

class Contracts(unittest.TestCase):
 def test_owned_source_and_real_training_source(self):
  self.assertIsNone(c.TRAINING_SOURCE_SHA)
  c.verify_training_binding(TRAIN,c.sha(TRAIN/'source_manifest.json'))
  self.assertEqual(c.verify_owned_sources(),c.sha(ROOT/'source_manifest.json'))
 def state(self):
  return {'visual.a':torch.zeros(2,dtype=torch.float32),'visual.b':torch.ones(3,dtype=torch.float32),'classifier.linear.weight':torch.zeros(750,512),'classifier.linear.bias':torch.zeros(750)}
 def test_joint_complete_head(self):
  r=c.validate_state(self.state(),arm='B25',visual_count=2,visual_numel=5)
  self.assertEqual(r['head_tensors'],2);self.assertEqual(r['trainable_numel'],384755)
 def test_joint_missing_head_rejected(self):
  s=self.state();del s['classifier.linear.bias']
  with self.assertRaises(ValueError):c.validate_state(s,arm='B50',visual_count=2,visual_numel=5)
 def test_nonfinite_or_wrong_dtype_rejected(self):
  for mode in ('nan','float16'):
   s=self.state();s['visual.a']=torch.tensor([float('nan'),0.]) if mode=='nan' else s['visual.a'].half()
   with self.assertRaises(ValueError):c.validate_state(s,arm='B25',visual_count=2,visual_numel=5)
 def rng(self):
  raw=np.random.RandomState(20261010).get_state();return dict(python_rng_state=random.Random(20261010).getstate(),numpy_rng_state=(raw[0],raw[1].tolist(),raw[2],raw[3],raw[4]),torch_rng_state=torch.Generator().manual_seed(20261010).get_state(),cuda_rng_state_all=[torch.ones(32,dtype=torch.uint8)])
 def test_private_rng_restore_without_global_change(self):
  py=random.getstate();cpu=torch.get_rng_state().clone();np_before=np.random.get_state();r=c.validate_rng_state(self.rng())
  self.assertTrue(r['torch_rng_private_CPU_restore']);self.assertFalse(r['cuda_accessed']);self.assertEqual(py,random.getstate());self.assertTrue(torch.equal(cpu,torch.get_rng_state()));np_after=np.random.get_state();self.assertTrue(np.array_equal(np_before[1],np_after[1]))
 def test_rng_missing_rejected(self):
  for key in self.rng():
   s=self.rng();del s[key]
   with self.assertRaises(ValueError):c.validate_rng_state(s)
 def test_rng_corrupt_rejected(self):
  for key,value in [('python_rng_state',()),('numpy_rng_state',('MT19937',[0],0,0,0.)),('torch_rng_state',torch.zeros(3)),('cuda_rng_state_all',[])]:
   s=self.rng();s[key]=value
   with self.assertRaises((ValueError,IndexError)):c.validate_rng_state(s)
 def optimizer(self):
  s={'trainable_state':self.state(),'scheduler':dict(T_max=20,last_epoch=20,base_lrs=[1.75e-6,5e-6]),'scaler':dict(scale=128.),'confidence_weighting':dict(epochs_updated=20)}
  s['optimizer']=dict(param_groups=[dict(params=[0,1],betas=(.9,.999),eps=1e-8,weight_decay=.01,initial_lr=1.75e-6,lr=0.),dict(params=[2,3],betas=(.9,.999),eps=1e-8,weight_decay=.01,initial_lr=5e-6,lr=0.)],state={i:dict(step=torch.tensor(3.),exp_avg=torch.zeros_like(v),exp_avg_sq=torch.zeros_like(v)) for i,v in enumerate(s['trainable_state'].values())});return s
 def test_optimizer_head_and_scheduler(self):
  s=self.optimizer();r=c.validate_optimizer_state(s,list(s['trainable_state']),visual_names=['visual.a','visual.b'],arm='B25');self.assertEqual(r['optimizer_parameter_groups'],[2,2])
  s['scheduler']['last_epoch']=19
  with self.assertRaises(ValueError):c.validate_optimizer_state(s,list(s['trainable_state']),visual_names=['visual.a','visual.b'],arm='B25')
 def test_optimizer_original_hyperparameters(self):
  for field,value in [('weight_decay',.02),('initial_lr',5e-5),('betas',(.8,.999)),('lr',1e-5)]:
   s=self.optimizer();s['optimizer']['param_groups'][0][field]=value
   with self.assertRaises(ValueError):c.validate_optimizer_state(s,list(s['trainable_state']),visual_names=['visual.a','visual.b'],arm='B25')
 def test_optimizer_missing_head_moment(self):
  s=self.optimizer();del s['optimizer']['state'][3]
  with self.assertRaises(ValueError):c.validate_optimizer_state(s,list(s['trainable_state']),visual_names=['visual.a','visual.b'],arm='B50')
 def test_real_protocol_rebound_queries_unchanged(self):
  _,parts=c.confirmation_rows();rows=parts['validation'];identity=json.loads((ROOT/'inputs/legacy_control/J5/input_identity.json').read_text(encoding='utf-8'))
  ctrl=dict(targets=np.array([int(r['target']) for r in rows]),groups=[r['group'] for r in rows],ids=[r['sample_id'] for r in rows],scores=np.empty((0,750)),identity=identity)
  queries,qi=load_queries(ctrl,ROOT/'inputs/frozen_queries');self.assertEqual(len(queries['natural_indices']),29737);self.assertEqual(len(queries['balanced_indices']),14760);self.assertEqual(len(queries['balanced_columns']),738)
  pred,proof=imported_graph_predictions(queries);self.assertTrue(proof['legacy_source_identity_preserved']);self.assertFalse(proof['new_source_identity_not_claimed'] is False)
  self.assertEqual(pred['balanced_supported_unique'].shape,(14760,));self.assertEqual(qi['selection_protocol_sha256'],c.sha(ROOT/'selection_protocol.json'))
 def test_primary_uses_imported_graph_control(self):
  scores=np.array([[6.,0.],[0.,6.],[5.,0.],[0.,5.]],dtype=np.float32);base=dict(scores=scores,targets=np.array([0,1,0,1]),train_counts=np.array([2,2]),ids=['a','b','c','d'],groups=['a','b','c','d'])
  q=dict(natural_indices=np.arange(4),natural_columns=np.arange(2),balanced_indices=np.arange(4),balanced_columns=np.arange(2));override={name:np.array([1,0,1,0],dtype=np.int64) for name in ['natural_unique_images','balanced_supported_unique']}
  row=compare_arrays(base,base,q,draws=0,only_protocol='balanced_supported_unique',control_predictions=override)[0];self.assertTrue(np.array_equal(row['predictions']['control_p7'],override['balanced_supported_unique']));self.assertGreater(row['p7']['candidate_vs_control']['delta_vs_baseline_pp'],0)
 def test_graph_ln4_protection(self):
  scores=np.array([[8.,0.],[.1,0.],[0.,.1]],dtype=np.float32);features=np.array([[1.,0.],[1.,0.],[1.,0.]],dtype=np.float32);values,detail=graph_scores(scores,features);self.assertTrue(np.array_equal(values[0],scores[0]));self.assertEqual(values.shape,scores.shape)
 def test_feature_collector_four_calls_no_extra_forward(self):
  class V(torch.nn.Module):
   def __init__(self):super().__init__();self.calls=0
   def forward(self,x):self.calls+=1;return x
  v=V();collector=FourViewFeatureCollector(v,layout='four_calls_B')
  try:
   for x in [torch.nn.functional.pad(torch.eye(2),(0,510))]*4:v(x)
   features=collector.finish(2);self.assertEqual(v.calls,4);self.assertEqual(features.shape,(2,512));self.assertTrue(torch.allclose(features.norm(dim=1),torch.ones(2)))
  finally:collector.close()
 def test_real_original_producer_signature_and_parser(self):
  result=dependency_probe(TRAIN,c.sha(TRAIN/'source_manifest.json'));self.assertTrue(result['real_vendor_cli_help_passed']);self.assertFalse(result['model_loaded']);self.assertFalse(result['checkpoint_opened']);self.assertFalse(result['images_opened']);self.assertEqual(result['training_steps'],0)
 def test_source_cli_help_entries(self):
  for name in ['run_exploration_diagnostic.py','compare_graph_vs_J5.py','bind_diagnostic.py']:
   child=subprocess.run([sys.executable,'-B',str(ROOT/name),'--help'],capture_output=True,text=True,env=dict(os.environ),timeout=60);self.assertEqual(child.returncode,0,child.stderr)
if __name__=='__main__':unittest.main(verbosity=2)

"""CPU failure/integration tests; no official backbone, image, server, or CUDA use."""
import copy,hashlib,importlib.util,json,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import yaml
ROOT=Path(__file__).resolve().parents[1]
BINDING=json.loads((ROOT/'tests/fixture_binding.json').read_text());TRAIN=Path(BINDING['training_root'])
sys.path[:0]=[str(ROOT),str(TRAIN/'src'),str(TRAIN/'scripts')]
from confirmation_diag_contract import *
from compare_confirmation import build_queries,load_queries,compare_arrays,eligibility
torch.set_num_threads(4)
def tiny(joint=True):
 s={'visual.weight':torch.eye(2)}
 if joint:s.update({'classifier.linear.weight':torch.arange(6).float().reshape(3,2),'classifier.linear.bias':torch.tensor([1.,2.,3.])})
 return s
class Head(nn.Module):
 def __init__(self):
  super().__init__();self.linear=nn.Linear(2,3)
  with torch.no_grad():self.linear.weight.zero_();self.linear.bias.zero_()
  for p in self.parameters():p.requires_grad_(False)
 def forward(self,x):return self.linear(x)
class Tests(unittest.TestCase):
 def test_real_dependency_entry_generated_production_interfaces_and_cli(self):
  command=[sys.executable,str(ROOT/'run_confirmation_diagnostic.py'),'--training-root',str(TRAIN),
   '--expected-training-source-sha256',BINDING['training_source_sha256'],'--dependency-check']
  r=subprocess.run(command,capture_output=True,text=True,timeout=60);self.assertEqual(r.returncode,0,r.stderr)
  d=json.loads(r.stdout);self.assertTrue(d['real_vendor_cli_help_passed']);self.assertTrue(d['real_vendor_production_argv_parsed'])
  self.assertTrue(d['imports_and_signatures_verified']);self.assertEqual(d['frozen_training_source_sha256'],BINDING['training_source_sha256'])
  for key in ('gpu_probed','images_opened','checkpoint_opened','remote_assets_opened','model_loaded'):self.assertFalse(d[key])
  for key in ('p1_contract','mainline_h1_contract','aictrack1.engine.training_queue'):
   self.assertTrue(Path(d['static_imports'][key]).is_relative_to(TRAIN))
 def test_actual_binding_requires_fresh_source_digest_and_positive_rows(self):
  a,m=verify_training_binding(TRAIN,BINDING['training_source_sha256']);self.assertEqual(a['h1_row_binding'],ROW_BINDING)
  self.assertEqual(m['selected_arm'],'J5')
  with self.assertRaises(ValueError):verify_training_binding(TRAIN,'0'*64)
  with self.assertRaises(ValueError):verify_training_binding(ROOT.parent/'training_source',sha(ROOT.parent/'training_source/source_manifest.json'))
 def test_C_frozen152_J_joint154_policy_finite_exact_shapes(self):
  for arm in ('C','J5','J20'):
   s=tiny(arm!='C');r=validate_state(s,arm=arm,visual_names=['visual.weight'],visual_count=1,visual_numel=4,num_classes=3,feature_dim=2)
   self.assertEqual(r['trainable_tensors'],1 if arm=='C' else 3)
  for s,arm in ((tiny(True),'C'),(tiny(False),'J5')):
   with self.assertRaises(ValueError):validate_state(s,arm=arm,visual_count=1,visual_numel=4,num_classes=3,feature_dim=2)
  bad=tiny();bad['classifier.linear.bias'][0]=float('nan')
  with self.assertRaises(ValueError):validate_state(bad,arm='J5',visual_count=1,visual_numel=4,num_classes=3,feature_dim=2)
 def test_actual_fv2_loader_frozen_H1_C_and_joint_J_override(self):
  folder=ROOT/'tests/fixtures/loader';folder.mkdir(parents=True,exist_ok=True)
  clip=folder/'toy_clip.pt';clip.write_bytes(b'only synthetic CPU backbone boundary')
  cfg=folder/'model.yaml';cfg.write_text(yaml.safe_dump(dict(model=dict(provider='OpenAI',name='ViT-B/32',weight_path=str(clip),expected_weight_sha256=sha(clip),runtime_downloads_allowed=False))))
  config=folder/'joint.yaml';checkpoint=folder/'last.pt'
  config.write_text(yaml.safe_dump(dict(experiment=dict(run_id='synthetic',model_config='model.yaml',head_checkpoint='head.pt',peft={'mode':'full_visual','lora_blocks':8},training={'epochs':20},visual_input_resolution=336,position_interpolation='bicubic_7x7_to_10x10',num_classes=3))))
  spec=importlib.util.spec_from_file_location('aictrack1.models.confirm_cpu_loader',ROOT/'reference/fv2_single_model.py');loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=loader;spec.loader.exec_module(loader)
  def backbone(*a):return SimpleNamespace(visual=nn.Linear(2,2,bias=False)),None,SimpleNamespace(weight_sha256=sha(clip),feature_dim=2)
  def configure(v,*a,**k):
   for p in v.parameters():p.requires_grad_(True)
  def head(*a,**k):return Head(),{'feature_cache_metadata':{'normalized':True}}
  state=dict(config_sha256=sha(config),mode='full_visual',visual_input_resolution=336,position_interpolation='bicubic_7x7_to_10x10',class_mapping_sha256='toy',num_classes=3,clip_weight_sha256=sha(clip))
  kwargs=dict(config_path=config,checkpoint_path=checkpoint,class_mapping_digest='toy',num_classes=3,split_sha256='toy',device=torch.device('cpu'),git_commit='synthetic',checkpoint_role='final')
  with patch.object(loader,'load_clip_vit_b32',backbone),patch.object(loader,'configure_visual_peft',configure),patch.object(loader,'enable_interpolated_visual_resolution',lambda *a,**k:None),patch.object(loader,'load_classifier_head_checkpoint',head):
   for joint in (False,True):
    state['trainable_state']=tiny(joint);torch.save(state,checkpoint);fitted=loader.load_diagnostic_model(**kwargs)
    expected=state['trainable_state']['classifier.linear.weight'] if joint else torch.zeros(3,2)
    self.assertTrue(torch.equal(fitted.model.classifier.linear.weight,expected));self.assertFalse(fitted.model.training)
    self.assertTrue(all(not p.requires_grad for p in fitted.model.parameters()))
   state['trainable_state'].pop('classifier.linear.bias');torch.save(state,checkpoint)
   with self.assertRaises(ValueError):loader.load_diagnostic_model(**kwargs)
 def test_actual_optimizer_C_onegroup_J_twogroups_and_terminal_rejection(self):
  for arm in ('C','J5'):
   s=tiny(arm!='C');names=list(s);groups=[dict(params=[0])]+([] if arm=='C' else [dict(params=[1,2])])
   opt=dict(param_groups=groups,state={i:dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(v),exp_avg_sq=torch.zeros_like(v)) for i,v in enumerate(s.values())})
   state=dict(trainable_state=s,optimizer=opt,scheduler=dict(T_max=20,last_epoch=20),scaler=dict(scale=65536.),confidence_weighting=dict(epochs_updated=20))
   validate_optimizer_state(state,names,visual_names=['visual.weight'],arm=arm)
   state['scheduler']['last_epoch']=8
   with self.assertRaises(ValueError):validate_optimizer_state(state,names,visual_names=['visual.weight'],arm=arm)
 def test_actual_complete_AC_restoration_rejects_row_hyperparameter_finite_and_coverage(self):
  from aictrack1.methods.confidence import ClasswiseHistoryWeighting
  obj=ClasswiseHistoryWeighting(torch.tensor([1,3]),torch.tensor([0,1]));obj.epochs_updated=20;obj.seen.fill_(1)
  state=obj.state_dict();validate_ac_state(state,[1,3],[0,1],{})
  for key,v in [('sample_indices',torch.tensor([1,4])),('class_ids',torch.tensor([1,0])),
       ('momentum',.1),('weights',torch.tensor([1.,float('nan')])),('seen',torch.tensor([1,0])),('epochs_updated',19)]:
   bad=dict(state);bad[key]=v
   with self.assertRaises(ValueError):validate_ac_state(bad,[1,3],[0,1],{})
 def test_fresh_metadata_rejects_oldsplit_best_wrongH1_partial(self):
  folder=ROOT/'tests/fixtures';head=folder/'head_bytes.pt';head.write_bytes(b'synthetic H1 binding bytes');hsha=sha(head)
  history=[dict(epoch=i,validation={'all':{'top1':.5}}) for i in range(1,21)]
  state=dict(competition_stage='stage2',training_scope='fixed_split_validation',mode='full_visual',visual_input_resolution=336,position_interpolation='bicubic_7x7_to_10x10',epoch=20,config_sha256='1'*64,train_manifest_sha256=MANIFEST_SHA,split_sha256=SPLIT_SHA,class_mapping_sha256=MAPPING_SHA,clip_weight_sha256=CLIP_SHA,num_classes=750,history=history,token_readout={'mode':'none'},head_initializer_sha256=hsha)
  metrics=dict(last_checkpoint_sha256='2'*64,training_scope='fixed_split_validation',train_count=118958,validation_count=29737,early_stop=None,history=history)
  exp=dict(head_checkpoint=str(head),selection_rule='fixed_final',token_readout={'mode':'none'},training=dict(epochs=20,seed=20261008,visual_learning_rate=1.75e-6,head_learning_rate=5e-6))
  kw=dict(arm='J5',config_sha='1'*64,last_sha='2'*64,registered=dict(assets=dict(h1=dict(sha256=hsha))))
  validate_checkpoint_metadata(state,metrics,exp,**kw)
  for key,v in [('split_sha256',OLD_SPLIT_SHA),('epoch',8),('head_initializer_sha256','0'*64),('training_scope','full_train_fixed_epochs')]:
   corrupt=dict(state);corrupt[key]=v
   with self.assertRaises(ValueError):validate_checkpoint_metadata(corrupt,metrics,exp,**kw)
  with self.assertRaises(ValueError):validate_checkpoint_metadata(state,dict(metrics,last_checkpoint_sha256='best'),exp,**kw)
 def test_supportonly_fresh_queries_exact_unique_and_cache_independent(self):
  _,parts=confirmation_rows();rows=parts['validation'];ids=[r['sample_id'] for r in rows]
  control=dict(ids=ids,groups=[r['group'] for r in rows],targets=np.array([int(r['target']) for r in rows]),scores=np.empty((0,750)),identity=dict(sample_ids_order_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest()))
  arrays,identity=load_queries(control,ROOT/'inputs/frozen_queries');self.assertEqual(len(arrays['balanced_columns']),738)
  indices=arrays['balanced_indices'];self.assertEqual(len(indices),14760);self.assertEqual(len({control['groups'][i] for i in indices}),14760)
  self.assertEqual(len(set(indices)),14760);self.assertFalse(identity['selection_uses_predictions'])
  self.assertEqual(identity['split_sha256'],SPLIT_SHA)
 def test_pair_rejects_unmatched_currentH1_source_runtime_oldsplit(self):
  base=dict(model_arm='C',selected_arm='J5',split_sha256=SPLIT_SHA,manifest_sha256=MANIFEST_SHA,class_mapping_sha256=MAPPING_SHA,h1_sha256='1'*64,view_protocol=VIEW_PROTOCOL,training_source_manifest_sha256='2'*64,diagnostic_source_manifest_sha256='3'*64,selection_receipt_sha256='4'*64,source_h1_manifest_sha256='5'*64,classifier_head_source='fresh_fold_h1_frozen',trainable_tensors=152,runtime=dict(torch_num_threads=4,torch_num_interop_threads=16))
  c=dict(identity=base,ids=['a'],groups=['g'],targets=np.array([0]),train_counts=np.array([1]))
  j=dict(c,identity=dict(base,model_arm='J5',classifier_head_source='joint_fixed_last_state',trainable_tensors=154));validate_pair(j,c)
  for key,v in [('h1_sha256','wrong'),('training_source_manifest_sha256','wrong'),('split_sha256',OLD_SPLIT_SHA),('runtime',dict(torch_num_threads=64))]:
   bad=dict(j,identity=dict(j['identity'],**{key:v}))
   with self.assertRaises(ValueError):validate_pair(bad,c)
 def test_frozen_P7_selfcomparison_determinism_and_no_added_natural_gate(self):
  rng=np.random.default_rng(11);targets=np.repeat(np.arange(3),20);scores=rng.normal(size=(60,3)).astype('float32')
  c=dict(scores=scores,targets=targets,groups=[str(i) for i in range(60)],ids=[str(i) for i in range(60)],train_counts=np.array([100,200,50]))
  q=build_queries(c);result=compare_arrays(c,c,q,draws=10)
  self.assertTrue(all(r['p7']['candidate_vs_control']['delta_vs_baseline_pp']==0 for r in result))
  result[1]['p7']['candidate_vs_control']['delta_vs_baseline_pp']=.1;result[0]['p7']['candidate_vs_control']['delta_vs_baseline_pp']=-.1
  e=eligibility(result);self.assertEqual(e['status'],'fresh_registered_direction_positive');self.assertFalse(e['natural_nonnegative_hard_gate']);self.assertFalse(e['ci_lower_bound_positive_hard_gate'])

class FeatureTests(unittest.TestCase):
    def test_features_hook_four_B_two_2B_order_zero_extra_forward_unchanged_logits(self):
        from view_features import FourViewFeatureCollector
        from aictrack1.models.peft import PEFTImageClassifier
        from aictrack1.evaluation.four_view import fixed_four_view_logits
        from torch.nn.functional import normalize
        class Visual(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, images):
                self.calls += 1
                return torch.stack([images[:, 0, 0, 0] + 2., images[:, 0, 0, -1] + 3., images.mean((1, 2, 3)) + 4.], dim=1).half()
        class Classifier(nn.Module):
            def forward(self, features):
                return torch.stack([features[:, 0] * 2. + features[:, 1], features[:, 2] - features[:, 0]], dim=1)
        visual = Visual()
        model = PEFTImageClassifier(visual, Classifier(), normalize_features=True).eval()
        native = torch.tensor([[[[1., 2., 3.]]], [[[4., 5., 6.]]]])
        scaled = torch.tensor([[[[7., 8., 9.]]], [[[10., 11., 12.]]]])
        baseline = fixed_four_view_logits(model, native, scaled)
        before = visual.calls
        collector = FourViewFeatureCollector(visual, feature_dim=3, layout='four_calls_B')
        observed = fixed_four_view_logits(model, native, scaled)
        features = collector.finish(2)
        self.assertEqual(visual.calls - before, 4)
        self.assertEqual(collector.calls, 4)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, observed, strict=True)))
        expected = []
        for images in [native, torch.flip(native, (-1,)), scaled, torch.flip(scaled, (-1,))]:
            raw = torch.stack([images[:, 0, 0, 0] + 2., images[:, 0, 0, -1] + 3., images.mean((1, 2, 3)) + 4.], dim=1).half().float()
            expected.append(normalize(raw, dim=-1))
        expected = normalize(torch.stack(expected).mean(0), dim=-1)
        self.assertTrue(torch.equal(features, expected))
        collector.close()
        before = visual.calls
        two = FourViewFeatureCollector(visual, feature_dim=3, layout='two_calls_2B')
        native_pair = model(torch.cat([native, torch.flip(native, (-1,))], dim=0))
        scaled_pair = model(torch.cat([scaled, torch.flip(scaled, (-1,))], dim=0))
        first, flipped = native_pair.chunk(2)
        resized, resized_flip = scaled_pair.chunk(2)
        combined = (first + flipped + resized + resized_flip) / 4
        self.assertEqual(visual.calls - before, 2)
        self.assertTrue(torch.equal(two.finish(2), expected))
        self.assertTrue(torch.equal(combined, baseline[2]))
        two.close()
        partial = FourViewFeatureCollector(visual, feature_dim=3)
        model(native)
        with self.assertRaises(ValueError):
            partial.finish(2)
        partial.close()


if __name__=='__main__':unittest.main(verbosity=2)

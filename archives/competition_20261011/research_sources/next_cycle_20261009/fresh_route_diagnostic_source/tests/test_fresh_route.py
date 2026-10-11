"""Science and real provider API/CLI CPU fixtures, no official checkpoint or image."""
import copy,hashlib,importlib.util,json,subprocess,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,torch,yaml
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
BINDING=json.loads((ROOT/'artifacts/cpu_fixture_binding.json').read_text());TRAIN=Path(BINDING['training_root'])
sys.path[:0]=[str(ROOT),str(TRAIN/'src'),str(TRAIN/'scripts')]
from fresh_diag_contract import *
from compare_fresh_route import build_queries,load_queries,compare_arrays,eligibility
from confirm_fixed_graph_vs_C import graph_scores,run_comparison,validate_features,load_existing_budget
from write_fresh_route_confirmation import direction,alias_sha
torch.set_num_threads(2)

def tiny(joint=False):
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
 def test_real_last_producer_keys_and_runtime_audit_statement_integration(self):
  import ast
  source=ast.parse((TRAIN/'scripts/train_n1_peft.py').read_text(encoding='utf-8'))
  nodes=[n for n in ast.walk(source) if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='checkpoint' and isinstance(n.value,ast.Dict)]
  self.assertEqual(len(nodes),1);keys={k.value for k in nodes[0].value.keys if isinstance(k,ast.Constant)}
  self.assertNotIn('head_initializer_sha256',keys)
  self.assertTrue({'fresh_route_source_manifest_sha256','fresh_route_arm','selected_route','classifier_head_source','classifier_head_restored_equal'}<=keys)
  wrapper=ast.parse((ROOT/'run_fresh_route_diagnostic.py').read_text())
  fn=next(n for n in wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='load_model')
  update=next(n for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(k.arg=='numel' for k in n.value.keywords))
  rtfile=ROOT/'artifacts/cpu_fixtures/runtime.json';rtfile.parent.mkdir(parents=True,exist_ok=True);rtfile.write_text('{}')
  ns=dict(audit={'trainable_numel':87849216},runtime={'runtime':{'torch_num_threads':4}},OUTPUT=rtfile.parent,sha=lambda p:'a'*64)
  exec(compile(ast.Module(body=[update],type_ignores=[]),'real_wrapper_runtime_update','exec'),ns)
  self.assertEqual(ns['audit']['runtime'],{'torch_num_threads':4})
 def test_actual_J5_GRAPH_provider_dependency_cli_not_naked_J_confirmation_gate(self):
  binding=json.loads((ROOT/'artifacts/cpu_fixture_binding_J5_GRAPH.json').read_text())
  r=subprocess.run([sys.executable,str(ROOT/'run_fresh_route_diagnostic.py'),'--training-root',binding['training_root'],'--expected-training-source-sha256',binding['training_source_sha256'],'--dependency-check'],capture_output=True,text=True,timeout=60)
  self.assertEqual(r.returncode,0,r.stderr);self.assertTrue(json.loads(r.stdout)['real_vendor_production_argv_parsed'])
 def test_actual_provider_C_S_binding_signatures_cli_no_model_gpu_or_images(self):
  args=[sys.executable,str(ROOT/'run_fresh_route_diagnostic.py'),'--training-root',str(TRAIN),'--expected-training-source-sha256',BINDING['training_source_sha256'],'--dependency-check']
  r=subprocess.run(args,capture_output=True,text=True,timeout=60);self.assertEqual(r.returncode,0,r.stderr)
  value=json.loads(r.stdout);self.assertTrue(value['real_vendor_cli_help_passed']);self.assertTrue(value['real_vendor_production_argv_parsed'])
  for key in ('gpu_probed','images_opened','checkpoint_opened','remote_assets_opened','model_loaded'):self.assertFalse(value[key])
  for name in ('p1_contract','mainline_h1_contract','aictrack1.engine.training_queue'):self.assertTrue(Path(value['static_imports'][name]).resolve().is_relative_to(TRAIN.resolve()))
  # Most important integration property: compatibility import really resolves the provider contract.
  script="import p1_contract,fresh_route_contract; from pathlib import Path; assert Path(fresh_route_contract.__file__).resolve().is_relative_to(Path("+repr(str(TRAIN))+")); print('provider_real_contract_PASS')"
  r=subprocess.run([sys.executable,'-c',script],env=__import__('os').environ|{'PYTHONPATH':str(TRAIN/'scripts')},capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
 def test_positive_actual_binding_rejects_oldsource_digest_and_reset_baseline(self):
  a,m=verify_training_binding(TRAIN,BINDING['training_source_sha256']);self.assertEqual(a['h1_row_binding'],ROW_BINDING);self.assertEqual(a['selected_arm'],'S')
  with self.assertRaises(ValueError):verify_training_binding(TRAIN,'0'*64)
  with self.assertRaises(ValueError):verify_training_binding(ROOT.parent/'training_source',sha(ROOT.parent/'training_source/source_manifest.json'))
 def test_complete_FP32_C_S152_J154_head_shape_finite_strict(self):
  for arm in ('C','S','J5'):
   state=tiny(arm=='J5');counts=validate_state(state,arm=arm,visual_names=['visual.weight'],visual_count=1,visual_numel=4,num_classes=3,feature_dim=2);self.assertEqual(counts['head_tensors'],2 if arm=='J5' else 0)
  for state,arm in ((tiny(True),'S'),(tiny(False),'J5')):
   with self.assertRaises(ValueError):validate_state(state,arm=arm,visual_count=1,visual_numel=4,num_classes=3,feature_dim=2)
  for value in (torch.eye(2).half(),torch.full((2,2),float('nan')),torch.ones(5)):
   with self.assertRaises(ValueError):validate_state({'visual.weight':value},arm='S',visual_count=1,visual_numel=4)
 def test_terminal_metadata_alpha_head_policy_source_partial_best_rejected(self):
  folder=ROOT/'artifacts/cpu_fixtures';folder.mkdir(parents=True,exist_ok=True);head=folder/'head_bytes.pt';head.write_bytes(b'synthetic head initializer bytes');hsha=sha(head)
  history=[dict(epoch=i,validation={'all':{'top1':.5}}) for i in range(1,21)]
  for arm,route in [('C','S'),('S','S'),('J5','J5_GRAPH')]:
   loss=dict(kind='sce',alpha=.5 if arm=='S' else .1,beta=1.,log_zero=-4.)
   state=dict(competition_stage='stage2',training_scope='fixed_split_validation',mode='full_visual',visual_input_resolution=336,position_interpolation='bicubic_7x7_to_10x10',epoch=20,config_sha256='1'*64,train_manifest_sha256=MANIFEST_SHA,split_sha256=SPLIT_SHA,class_mapping_sha256=MAPPING_SHA,clip_weight_sha256=CLIP_SHA,num_classes=750,history=history,token_readout={'mode':'none'},fresh_route_arm=arm,fresh_route_source_manifest_sha256='a'*64,selected_route=route,classifier_head_source='new_fold_H1_frozen' if arm!='J5' else 'joint_fixed_last_state',classifier_head_restored_equal=arm!='J5')
   metrics=dict(last_checkpoint_sha256='2'*64,training_scope='fixed_split_validation',train_count=118958,validation_count=29737,early_stop=None,history=history,head_initializer={'sha256':hsha},base_loss=loss,trainable_total_parameters=VISUAL_NUMEL if arm!='J5' else JOINT_NUMEL)
   exp=dict(head_checkpoint=str(head),selection_rule='fixed_final',token_readout={'mode':'none'},training=dict(epochs=20,seed=20261008,visual_learning_rate=1.75e-6,head_learning_rate=ARMS[arm]),base_loss=loss)
   args=dict(arm=arm,config_sha='1'*64,last_sha='2'*64,registered=dict(assets=dict(h1=dict(sha256=hsha)),training_source_manifest_sha256='a'*64,selected_route=route))
   validate_checkpoint_metadata(state,metrics,exp,**args)
   for key,value in [('split_sha256',OLD_SPLIT_SHA),('epoch',8),('fresh_route_arm','wrong'),('classifier_head_source','original_fold_H1_frozen'),('fresh_route_source_manifest_sha256','old')]:
    with self.assertRaises(ValueError):validate_checkpoint_metadata(dict(state,**{key:value}),metrics,exp,**args)
   with self.assertRaises(ValueError):validate_checkpoint_metadata(state,dict(metrics,last_checkpoint_sha256='best'),exp,**args)
   with self.assertRaises(ValueError):validate_checkpoint_metadata(state,metrics,dict(exp,base_loss=dict(loss,alpha=.2)),**args)
 def test_one_visual_optimizer_C_S_two_joint_J_and_full_AC_ID_labels(self):
  for arm in ARMS:
   tensors=tiny(arm=='J5');groups=[dict(params=[0])]+([dict(params=[1,2])] if arm=='J5' else [])
   state=dict(trainable_state=tensors,optimizer=dict(param_groups=groups,state={i:dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(v),exp_avg_sq=torch.zeros_like(v)) for i,v in enumerate(tensors.values())}),scheduler=dict(T_max=20,last_epoch=20),scaler=dict(scale=65536.),confidence_weighting=dict(epochs_updated=20))
   validate_optimizer_state(state,list(tensors),visual_names=['visual.weight'],arm=arm)
   with self.assertRaises(ValueError):validate_optimizer_state(dict(state,scheduler=dict(T_max=20,last_epoch=8)),list(tensors),visual_names=['visual.weight'],arm=arm)
  from aictrack1.methods.confidence import ClasswiseHistoryWeighting
  obj=ClasswiseHistoryWeighting(torch.tensor([1,3]),torch.tensor([0,1]));obj.epochs_updated=20;obj.seen.fill_(1);state=obj.state_dict();validate_ac_state(state,[1,3],[0,1],{})
  for key,value in [('sample_indices',torch.tensor([3,1])),('class_ids',torch.tensor([1,0])),('epochs_updated',19),('seen',torch.tensor([1,0])),('weights',torch.tensor([float('nan'),1.]))]:
   with self.assertRaises(ValueError):validate_ac_state(dict(state,**{key:value}),[1,3],[0,1],{})
 def test_real_new_query738_support_only_group_independence(self):
  _,parts=confirmation_rows();rows=parts['validation'];ids=[r['sample_id'] for r in rows]
  control=dict(ids=ids,groups=[r['group'] for r in rows],targets=np.array([int(r['target']) for r in rows]),scores=np.empty((0,750)),identity=dict(sample_ids_order_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest()))
  arrays,value=load_queries(control,ROOT/'inputs/frozen_queries');self.assertEqual(len(arrays['balanced_columns']),738);self.assertEqual(len(arrays['balanced_indices']),14760);self.assertEqual(len({control['groups'][i] for i in arrays['balanced_indices']}),14760);self.assertFalse(value['selection_uses_predictions'])
 def test_pair_C_S_or_J5_rejects_oldC_H1_source_runtime_reordered_rows(self):
  for route,arm in ROUTES.items():
   base=dict(model_arm='C',selected_route=route,split_sha256=SPLIT_SHA,manifest_sha256=MANIFEST_SHA,class_mapping_sha256=MAPPING_SHA,h1_sha256=H1_SHA,view_protocol=VIEW_PROTOCOL,training_source_manifest_sha256='a'*64,diagnostic_source_manifest_sha256='b'*64,admission_receipt_sha256='c'*64,source_h1_manifest_sha256=H1_SOURCE_SHA,classifier_head_source='new_fold_H1_frozen',trainable_tensors=152,runtime=dict(torch_num_threads=4,torch_num_interop_threads=16))
   control=dict(identity=base,ids=['a'],groups=['g'],targets=np.array([0]),train_counts=np.array([1]))
   candidate=dict(control,identity=dict(base,model_arm=arm,classifier_head_source='new_fold_H1_frozen' if arm=='S' else 'joint_fixed_last_state',trainable_tensors=152 if arm=='S' else 154));validate_pair(candidate,control)
   for key,v in [('h1_sha256','45266old'),('split_sha256',OLD_SPLIT_SHA),('runtime',dict(torch_num_threads=64)),('admission_receipt_sha256','wrong')]:
    with self.assertRaises(ValueError):validate_pair(dict(candidate,identity=dict(candidate['identity'],**{key:v})),control)
   with self.assertRaises(ValueError):validate_pair(dict(candidate,ids=['reordered']),control)
 def test_P7_self_deterministic_exact_decline_boundary_no_natural_CI_gate(self):
  targets=np.repeat(np.arange(3),20);scores=np.zeros((60,3),np.float32);scores[np.arange(60),targets]=2.
  control=dict(scores=scores,targets=targets,groups=[str(i) for i in range(60)],ids=[str(i) for i in range(60)],train_counts=np.array([100,200,50]));q=build_queries(control)
  rows=compare_arrays(control,control,q,draws=10);self.assertTrue(all(r['p7']['candidate_vs_control']['delta_vs_baseline_pp']==0 for r in rows))
  rows[1]['p7']['candidate_vs_control']['delta_vs_baseline_pp']=.1;rows[0]['p7']['candidate_vs_control']['delta_vs_baseline_pp']=-.5
  result=eligibility(rows);self.assertEqual(result['status'],'fresh_registered_direction_positive');self.assertFalse(result['natural_nonnegative_hard_gate']);self.assertFalse(result['ci_lower_bound_positive_hard_gate'])
 def test_exact_two_of_twenty_decline_report_does_not_round_away_minus10(self):
  targets=np.repeat(np.arange(3),20);scores=np.zeros((60,3),np.float32);scores[np.arange(60),targets]=4.
  control=dict(scores=scores,targets=targets,groups=[str(i) for i in range(60)],ids=[str(i) for i in range(60)],train_counts=np.array([100,200,50]))
  altered=scores.copy();altered[:2,0]=0;altered[:2,1]=4;candidate=dict(control,scores=altered)
  result=compare_arrays(candidate,control,build_queries(control),draws=10)
  for row in result:
   declined=next(c for c in row['p7_per_class_support_and_recall'] if c['original_class_index']==0)
   self.assertEqual(declined['candidate_correct'],18);self.assertEqual(declined['control_correct'],20);self.assertEqual(declined['delta_pp'],-10.)
   self.assertTrue(any(c['original_class_index']==0 for c in row['supported_large_declines']))
 def test_graph_singlepoint_T1_originalP_residual_reciprocal_protected_exact(self):
  from p3_graph import blocked_topk,mutual_graph,smooth
  x=np.array([[1.,0],[.99,.1],[0,1],[-1,0]],np.float32);x/=np.linalg.norm(x,axis=1,keepdims=True)
  scores=np.array([[4.,0,0],[.3,.2,0],[0,.2,.1],[0,0,.5]],np.float32)
  ns,ws,_=blocked_topk(x,20,row_block=2);w,detail,degree=mutual_graph(ns,ws,10);smoothed,data=smooth(scores,w,degree,.3)
  base=torch.softmax(torch.from_numpy(scores),1).numpy();q=base.copy();anchor=(scores.max(1)-np.partition(scores,-2,axis=1)[:,-2]>=np.log(4))|(degree==0)
  for _ in range(3):q=.7*base+.3*(w@q);q[anchor]=base[anchor]
  expected=np.log(q)+torch.logsumexp(torch.from_numpy(scores),1).numpy()[:,None];expected[anchor]=scores[anchor]
  self.assertTrue(np.allclose(smoothed,expected,atol=5e-7,rtol=0));self.assertTrue(np.array_equal(smoothed[anchor],scores[anchor]));self.assertEqual(data['protected_raw_prediction_changes'],0);self.assertTrue(detail['symmetric_adjacency'])
 def test_graph_direct_combo_C_and_increment_J5_are_separate_scientific_outputs(self):
  rng=np.random.default_rng(5);targets=np.repeat(np.arange(3),20);scores=rng.normal(size=(60,3)).astype('float32');features=rng.normal(size=(60,4)).astype('float32');features/=np.linalg.norm(features,axis=1,keepdims=True)
  control=dict(scores=scores,targets=targets,ids=[str(i) for i in range(60)],groups=[str(i) for i in range(60)],train_counts=np.array([80,100,40]));candidate=dict(control,scores=scores+.03*rng.normal(size=scores.shape).astype('float32'))
  raw,combo,inc,details=run_comparison(candidate,control,build_queries(control),features,draws=10)
  self.assertEqual([r['name'] for r in raw],[r['name'] for r in combo]);self.assertEqual(len(inc),2)
  for r,c,i in zip(raw,combo,inc,strict=True):self.assertAlmostEqual(c['p7']['candidate_vs_control']['delta_vs_baseline_pp'],r['p7']['candidate_vs_control']['delta_vs_baseline_pp']+i['p7']['candidate_vs_control']['delta_vs_baseline_pp'],places=5)
 def test_graph_budget_missing_or_wrongsource_never_creates_newcounter(self):
  with self.assertRaises(ValueError):load_existing_budget(ROOT/'artifacts/not_a_shared_graph_root')
 def test_typed_direction_positive_combo_can_pass_negative_rawJ_and_natural(self):
  row=lambda name,delta:dict(name=name,p7=dict(candidate_vs_control=dict(delta_vs_baseline_pp=delta,group_bootstrap=dict(ci95_pp=[-.5,.8]))))
  base=dict(complete=True,selected_route='J5_GRAPH',selected_arm='J5',p1_arm='J5',test_access=False,used_test_or_online_scores=False,split_sha256=SPLIT_SHA,h1_sha256=H1_SHA,source_h1_manifest_sha256=H1_SOURCE_SHA,selection_protocol_sha256=sha(ROOT/'selection_protocol.json'),frozen_postprocess_sha256=sha(ROOT/'frozen_postprocess.json'),environment=dict(cpu_threads=2),protocols=[row('natural_unique_images',-.5),row('balanced_supported_unique',.2)],eligibility=dict(primary_delta_pp=.2,natural_p7_delta_pp=-.5),schema='aictrack1.fresh_route.graph_combo_paired.v1',graph_source_sha256=__import__('confirm_fixed_graph_vs_C').GRAPH_SOURCE_SHA,fixed_graph_protocol_sha256=sha(ROOT/'fixed_graph_protocol.json'),fixed_graph=dict(k=10,lambda_value=.3,iterations=3,probability_temperature=1.,margin_protection=np.log(4.)),primary_scope='J5_graph_P7_vs_C',additional_model_forwards=0,model_updates=0,raw_j5_vs_C=[row('natural_unique_images',-.4),row('balanced_supported_unique',-.2)],graph_increment_vs_J5=[row('natural_unique_images',-.1),row('balanced_supported_unique',.4)])
  self.assertEqual(direction(base,'J5_GRAPH')['primary_delta_vs_C_pp'],.2)
  with self.assertRaises(ValueError):direction(dict(base,schema='aictrack1.fresh_route.raw_paired.v1'),'J5_GRAPH')
  with self.assertRaises(ValueError):direction(dict(base,split_sha256=OLD_SPLIT_SHA),'J5_GRAPH')
  broken=copy.deepcopy(base);broken['protocols'][1]['p7']['candidate_vs_control']['delta_vs_baseline_pp']=0
  with self.assertRaises(ValueError):direction(broken,'J5_GRAPH')
 def test_receipt_alias_conflict_and_cli_actual_help(self):
  with self.assertRaises(ValueError):alias_sha({'candidate':dict(identity_sha256='a'*64,provenance=dict(identity_sha256='b'*64))},'candidate')
  for name in ('compare_fresh_route.py','confirm_fixed_graph_vs_C.py','write_fresh_route_confirmation.py'):
   r=subprocess.run([sys.executable,str(ROOT/name),'--help'],capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)


class ActualLoader(unittest.TestCase):
 def test_actual_fv2_loader_frozen_H1_C_and_joint_J_override(self):
   folder=ROOT/'artifacts/cpu_fixtures/loader';folder.mkdir(parents=True,exist_ok=True)
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

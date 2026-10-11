"""CPU science and contract fixtures only; never real model/query/SSH/CUDA."""
import copy,json,math,os,random,shutil,subprocess,sys,time,unittest,uuid
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from torch import nn
import yaml
import fresh_route_contract as c
import fresh_route_runtime as rt
import prepare_fresh_route as prep
from aictrack1.models.heads import LinearClassifier
from aictrack1.models.peft import PEFTImageClassifier
from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict,configure_classifier_trainability_from_state
from aictrack1.methods.confidence import ClasswiseHistoryWeighting
from aictrack1.methods.confidence_observation import observation_protocol,validate_observation_resume
from aictrack1.engine.training_queue import QueueItem,run_serial_queue,read_training_lock
from train_n1_peft import _base_loss_per_sample,_weighted_base_loss

@contextmanager
def fixture():
    path=ROOT.parent/'operator_tools'/('fresh_route_cpu_fixture_'+uuid.uuid4().hex);path.mkdir(parents=True)
    try:yield path
    finally:
        if not path.resolve().is_relative_to((ROOT.parent/'operator_tools').resolve()):raise ValueError('fixture escape')
        shutil.rmtree(path)

def admission(tmp,route='J5_GRAPH',delta=.251):
    evidence={}
    for name in ['checkpoint','config','paired_report','graph_result','combo_report','graph_protocol']:
        path=tmp/(name+'.json');path.write_text(json.dumps({'toy_fixture':True,'name':name}))
        evidence[name]={'path':str(path.resolve()),'sha256':c.sha(path)}
    source=ROOT.parent/('supervision_source' if route=='S' else 'training_source')/'source_manifest.json'
    evidence['training_source']={'path':str(source),'sha256':c.sha(source)}
    review=tmp/'root_review.json';review.write_text(json.dumps({'selected_route':route,'decision':'proceed_to_fresh_confirmation','reason':'CPU fixture only'}))
    evidence['root_review']={'path':str(review.resolve()),'sha256':c.sha(review)}
    v=dict(schema='aictrack1.fresh_route.admission.v1',status='selected_recipe_for_fresh_route_confirmation',selected_route=route,selected_arm=c.ROUTE_ARM[route],
        test_access=False,used_test_or_online_scores=False,root_reviewed=True,selection_split_sha256=c.OLD_SPLIT_SHA,
        checkpoint_epoch=20,checkpoint_role='last',training_scope='fixed_split_validation',primary_scope='S_P7_vs_C' if route=='S' else 'J5_graph_P7_vs_C',
        primary_delta_vs_C_pp=delta,selected_checkpoint_sha256=evidence['checkpoint']['sha256'],selected_config_sha256=evidence['config']['sha256'],
        selected_training_source_manifest_sha256=evidence['training_source']['sha256'],selection_protocol_sha256='b60a39fc5e966b45f97a04bae30e5d2b67fe70b83f96c01c1d97799622cd2a02',evidence=evidence)
    if route=='J5_GRAPH':v.update(graph_parameters={'k':10,'lambda':.3,'iterations':3,'probability_temperature':1.,'protection_raw_margin':math.log(4)},raw_J5_P7_delta_vs_C_pp=.06784260516)
    return v

class Tests(unittest.TestCase):
    def test_exact_single_axis_configs_and_no_third_arm(self):
        base=json.loads((ROOT/'inputs/base_C_config.json').read_text())
        for route in c.ROUTE_ARM:
            p=prep.fresh_assets(route,'a'*64);control=c.expected_config('C',p,base);candidate=c.expected_config(p['selected_arm'],p,base)
            self.assertEqual(control['training']['epochs'],20);self.assertEqual(control['training']['seed'],20261008)
            self.assertEqual(control['training']['head_learning_rate'],0.);self.assertEqual(control['base_loss']['alpha'],.1)
            revised=copy.deepcopy(candidate);revised['run_id']=control['run_id'];revised['output_dir']=control['output_dir']
            if route=='S':revised['base_loss']['alpha']=.1
            else:revised['training']['head_learning_rate']=0.
            self.assertEqual(revised,control)
            for arm in ('C',p['selected_arm']):c.validate_config(c.expected_config(arm,p,base),arm,p)
            with self.assertRaises(ValueError):c.expected_config('J5' if route=='S' else 'S',p,base)

    def test_admission_graph_route_not_naked_j_gate(self):
        with fixture() as tmp:
            v=admission(tmp);self.assertEqual(c.verify_admission_evidence(v)['selected_route'],'J5_GRAPH')
            for mutation in [{'primary_delta_vs_C_pp':0.},{'primary_scope':'J5_P7_vs_C'},{'selected_arm':'S'},{'checkpoint_role':'best'},{'used_test_or_online_scores':True},{'selection_protocol_sha256':'b'*64}]:
                bad=copy.deepcopy(v);bad.update(mutation)
                with self.assertRaises(ValueError):c.validate_admission(bad)
            for key,value in [('k',20),('lambda',.1),('iterations',2),('probability_temperature',.75),('protection_raw_margin',2.)]:
                bad=copy.deepcopy(v);bad['graph_parameters'][key]=value
                with self.assertRaises(ValueError):c.validate_admission(bad)
            v=admission(tmp,'S',.099)
            with self.assertRaises(ValueError):c.validate_admission(v)
            v['primary_delta_vs_C_pp']=.1;c.verify_admission_evidence(v)

    def test_evidence_hash_root_review_and_alias_drift(self):
        with fixture() as tmp:
            v=admission(tmp);bad=copy.deepcopy(v);bad['selected_checkpoint_sha256']='b'*64
            with self.assertRaises(ValueError):c.validate_admission(bad)
            Path(v['evidence']['paired_report']['path']).write_text('drift')
            with self.assertRaises(ValueError):c.verify_admission_evidence(v)
            v=admission(tmp);review=Path(v['evidence']['root_review']['path']);review.write_text(json.dumps({'selected_route':'S','decision':'proceed_to_fresh_confirmation','reason':'toy'}));v['evidence']['root_review']['sha256']=c.sha(review)
            with self.assertRaises(ValueError):c.verify_admission_evidence(v)

    def test_real_accepted_h1_metadata_new_fold_not_full(self):
        p=prep.fresh_assets('S','a'*64);c.validate_assets(p)
        h=json.loads((ROOT/'inputs/h1_actual_CPU_complete.json').read_text())
        self.assertEqual(p['h1_row_binding'],h['producer_receipt']['row_binding']);self.assertEqual(p['assets']['h1']['sha256'],c.H1_SHA)
        for path,value in [('h1_training_scope','full_train_fixed_epochs'),('h1_validation_used_for_selection',True),('old_query_overlap',1),('source_h1_manifest_sha256','a'*64)]:
            bad=copy.deepcopy(p);bad[path]=value
            with self.assertRaises(ValueError):c.validate_assets(bad)
        bad=copy.deepcopy(p);bad['h1_row_binding']['train_rows_sha256']='a'*64
        with self.assertRaises(ValueError):c.validate_assets(bad)
        bad=copy.deepcopy(p);bad['assets']['h1']['sha256']='45266a98329a59f708b12f394775640e032e50be8c44077a1037048f28c88753'
        with self.assertRaises(ValueError):c.validate_assets(bad)

    def test_disk_resume_actual_loss_ac_rng_optimizer_head_policy(self):
        torch.set_num_threads(1)
        for arm,lr in c.ARMS.items():
            torch.manual_seed(3);random.seed(3);np.random.seed(3)
            model=PEFTImageClassifier(nn.Linear(5,8),LinearClassifier(8,4),normalize_features=True)
            for p in model.classifier.parameters():p.requires_grad_(lr>0)
            groups=[dict(params=list(model.visual.parameters()),lr=1.75e-6)]
            if lr:groups.append(dict(params=list(model.classifier.parameters()),lr=lr))
            opt=torch.optim.AdamW(groups,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20);scaler=torch.amp.GradScaler('cuda',enabled=False)
            ids=torch.arange(8);targets=ids%4;confidence=ClasswiseHistoryWeighting(ids,targets,momentum=.8,history_mix=.5,min_weight=.02,magnitude_mix=.75);protocol=observation_protocol('strong',20261008)
            def step():
                x=torch.randn(8,5)*(1.+random.random()*.01+float(np.random.rand())*.01);opt.zero_grad(set_to_none=True);logits=model(x)
                loss=_weighted_base_loss(_base_loss_per_sample(logits,targets,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.5 if arm=='S' else .1),confidence.weights_for(ids))
                scaler.scale(loss).backward();scaler.unscale_(opt)
                for p in model.classifier.parameters():
                    if lr:self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.norm()),0.)
                    else:self.assertIsNone(p.grad)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);scaler.step(opt);scaler.update();sched.step()
                ce=torch.nn.functional.cross_entropy(logits.detach(),targets,reduction='none');confidence.update_epoch(ids,ce,logits.detach().softmax(1)[ids,targets],logits.detach().argmax(1));return float(loss.detach())
            head0={k:v.detach().clone() for k,v in model.classifier.named_parameters()};step();rng=np.random.get_state();rng=(rng[0],rng[1].tolist(),*rng[2:])
            saved=dict(trainable_state=trainable_state_dict(model),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),confidence_weighting=confidence.state_dict(),confidence_observation_protocol=protocol,python_rng_state=random.getstate(),numpy_rng_state=rng,torch_rng_state=torch.random.get_rng_state(),epoch=1)
            with fixture() as tmp:
                torch.save(saved,tmp/'last.pt');del saved;expected_loss=step();expected=trainable_state_dict(model);expected_conf=confidence.state_dict();expected_opt=copy.deepcopy(opt.state_dict());expected_sched=copy.deepcopy(sched.state_dict())
                state=torch.load(tmp/'last.pt',weights_only=True,map_location='cpu');validate_observation_resume(state,protocol);load_trainable_state_dict(model,state['trainable_state']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler']);confidence.load_state_dict(state['confidence_weighting']);random.setstate(state['python_rng_state']);np.random.set_state(state['numpy_rng_state']);torch.random.set_rng_state(state['torch_rng_state'])
                self.assertEqual(step(),expected_loss)
                for name,v in trainable_state_dict(model).items():self.assertTrue(torch.equal(v,expected[name]),name)
                for name,v in confidence.state_dict().items():self.assertTrue(torch.equal(v,expected_conf[name]) if isinstance(v,torch.Tensor) else v==expected_conf[name])
                for i,values in opt.state_dict()['state'].items():
                    for key,v in values.items():self.assertTrue(torch.equal(v,expected_opt['state'][i][key]))
                self.assertEqual(sched.state_dict(),expected_sched);self.assertEqual(sched.T_max,20)
            self.assertEqual({x for x in expected if x.startswith('classifier.')},{'classifier.linear.weight','classifier.linear.bias'} if lr else set())
            self.assertEqual(any(not torch.equal(v,head0[k]) for k,v in model.classifier.named_parameters()),lr>0)
            for p in model.classifier.parameters():p.requires_grad_(False)
            self.assertEqual(configure_classifier_trainability_from_state(model.classifier,expected),lr>0)

    def test_trainable_count_meta_and_frozen_head_assertions(self):
        # Count/shape fixture does not allocate the 350MB visual backbone or call a forward.
        class CountModel:
            def __init__(self,joint):
                self.head=nn.Linear(512,750);self.values=[('visual.big',nn.Parameter(torch.empty(87849216-151,device='meta')))]
                self.values += [(f'visual.p{i}',nn.Parameter(torch.empty(1,device='meta'))) for i in range(151)]
                for p in self.head.parameters():p.requires_grad_(joint)
                self.values += [('classifier.'+k,p) for k,p in self.head.named_parameters()]
            def named_parameters(self):return iter(self.values)
        for joint in (False,True):
            model=CountModel(joint);reference={k:v.detach().clone() for k,v in model.head.state_dict().items()};raw={'training':{'head_learning_rate':5e-6 if joint else 0.}}
            rt.assert_route_head(model,model.head,reference,raw)
            if not joint:
                with torch.no_grad():model.head.bias[0]+=1
                with self.assertRaises(RuntimeError):rt.assert_route_head(model,model.head,reference,raw)
            else:
                model.head.weight.requires_grad_(False)
                with self.assertRaises(RuntimeError):rt.assert_route_head(model,model.head,reference,raw)

    def test_original_budget_and_matched_runtime_no_reset(self):
        rt.validate_runtime({'environment':rt.RUNTIME_ENV,'actual_torch_threads':4})
        with self.assertRaises(RuntimeError):rt.validate_runtime({'environment':rt.RUNTIME_ENV,'actual_torch_threads':64})
        with fixture() as tmp,patch.object(rt.time,'time',return_value=1791499000):
            kwargs=dict(start=1791497972.440027,deadline=1791526772.440027,initial=13013950464,diskmax=8589934592,minimum=3221225472)
            expected=rt.bind_budget(tmp,**kwargs);self.assertEqual(rt.bind_budget(tmp,**kwargs),expected)
            for key in ['start','deadline','initial']:
                bad=kwargs|{key:kwargs[key]+1}
                with self.assertRaises(ValueError):rt.bind_budget(tmp,**bad)
            with self.assertRaises(ValueError):rt.bind_budget(tmp,**(kwargs|{'diskmax':8589934593}))

    def test_clip_coefficients_amp_accounting(self):
        v=rt.clip_diagnostics(torch.tensor([.5,2.,float('inf')]),1.,7,2)
        self.assertEqual(v['clip_coefficient_nonfinite_excluded'],1)
        self.assertEqual(v['clip_coefficient_finite_norm_count'],2)
        self.assertEqual(v['amp_skipped_optimizer_updates'],2);self.assertEqual(v['amp_successful_optimizer_updates'],7)
        self.assertAlmostEqual(v['clip_coefficient_quantiles']['0.0'],.5,places=5);self.assertEqual(v['clip_coefficient_quantiles']['1.0'],1.)

    def test_prepare_new_bundle_exact_two_configs_preserves_template(self):
        with fixture() as tmp:
            originals={rel:c.sha(ROOT/rel) for rel in prep.verify_template()['files']}
            for route in c.ROUTE_ARM:
                folder=tmp/route;folder.mkdir();v=admission(folder,route);path=folder/'admission.json';path.write_text(json.dumps(v));target=folder/'bound'
                receipt=prep.prepare(target,path,c.sha(path));self.assertEqual(receipt['selected_route'],route);self.assertEqual(c.verify_files(target)['status'],'bound')
                self.assertEqual({x.name for x in (target/'configs/experiment').glob('*.yaml')},{'p1_C.yaml',f'p1_{c.ROUTE_ARM[route]}.yaml'})
                m=json.loads((target/'source_manifest.json').read_text());self.assertEqual(m['h1_sha256'],c.H1_SHA);self.assertEqual(m['source_h1_manifest_sha256'],c.H1_SOURCE_SHA)
                self.assertEqual(json.loads((target/'inputs/p1_assets.json').read_text())['h1_row_binding'],json.loads((ROOT/'inputs/h1_actual_CPU_complete.json').read_text())['producer_receipt']['row_binding'])
                with self.assertRaises(FileExistsError):prep.prepare(target,path,c.sha(path))
                (target/'scripts/p1_contract.py').write_text('drift')
                with self.assertRaises(ValueError):c.verify_files(target)
            self.assertEqual(originals,{rel:c.sha(ROOT/rel) for rel in originals})

    def test_actual_cli_help_and_imports_no_gpu(self):
        for script in ['prepare_fresh_route.py','run_fresh_route_arm.py','run_fresh_route_pair.py','train_n1_peft.py']:
            done=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],capture_output=True,text=True,timeout=30)
            self.assertEqual(done.returncode,0,script+done.stderr);self.assertIn('usage:',done.stdout)

    def test_canonical_queue_real_exact_resume_lease(self):
        with fixture() as root:
            (root/'scripts').mkdir();(root/'scripts/train_n1_peft.py').write_text('# CPU fixture\n');out=root/'artifacts/experiments/C';out.mkdir(parents=True);(out/'last.pt').write_bytes(b'toy')
            lock=root/'canonical.lock.json'
            def launch(command,cwd,handle,on_start):
                self.assertEqual(command[-2],'--resume');self.assertEqual(Path(command[-1]),out/'last.pt');self.assertIsNotNone(read_training_lock(lock));on_start(12345);self.assertEqual(read_training_lock(lock).trainer_pid,12345)
                (out/'config.yaml').write_text('fixture');(out/'command.txt').write_text('fixture');(out/'final.pt').write_bytes(b'fixture');(out/'metrics.json').write_text(json.dumps({'selected_checkpoint':'final.pt'}));return 0
            item=QueueItem('C','config.yaml','artifacts/experiments/C','fixture','scripts/train_n1_peft.py','artifacts/experiments/C/last.pt')
            self.assertEqual(run_serial_queue(root=root,python=Path(sys.executable),logs=root/'logs',queue_name='fresh-cpu-fixture',items=[item],process_probe=lambda:[],launcher=launch,lock_path=lock),0);self.assertFalse(lock.exists())

if __name__=='__main__':
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt=dict(schema='aictrack1.fresh_route.training.cpu_tests.v1',status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.monotonic()-started,python=sys.executable,torch=torch.__version__,cuda_used=False,server_access=False,actual_fresh_training_run=False,template_manifest_sha256=c.sha(ROOT/'template_manifest.json'))
    path=ROOT.parent/'operator_tools/fresh_route_training_cpu_test_receipt.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');raise SystemExit(not result.wasSuccessful())

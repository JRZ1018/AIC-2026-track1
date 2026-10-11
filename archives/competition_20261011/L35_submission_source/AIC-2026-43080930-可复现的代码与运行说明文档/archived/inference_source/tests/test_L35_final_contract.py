"""Independent CPU fixtures for learned-head, view, corruption and package contracts."""
from __future__ import annotations
import copy, hashlib, json, os, subprocess, sys, unittest, zipfile, uuid, time
from unittest.mock import patch,Mock
import numpy as np
from contextlib import contextmanager
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from torch import nn
import final_contract as fc
from aictrack1.engine.peft import trainable_state_dict
from aictrack1.models.heads import LinearClassifier as LinearClassifierHead
from aictrack1.models.peft import PEFTImageClassifier
from aictrack1.evaluation.four_view import fixed_four_view_logits
from aictrack1.evaluation.balanced_prior import fit_uniform_soft_bias
from aictrack1.data.class_mapping import ClassEntry,ClassMapping
from aictrack1.submission import SubmissionSpec,build_submission_package,validate_submission_csv,validate_submission_zip
from view_features import FourViewFeatureCollector
from accept_return import verify_return_files

@contextmanager
def fixture_directory():
    # Windows managed sandbox denies mkdtemp's private ACL. Ordinary new folders
    # are kept as bounded evidence; never recursively delete a computed path.
    folder=ROOT/'artifacts'/'test_fixtures'/('final_route_cpu_fixture_'+uuid.uuid4().hex)
    folder.mkdir(parents=True,exist_ok=False)
    yield str(folder)

def toy():
    visual=nn.Linear(3,512);head=LinearClassifierHead(512,750)
    model=PEFTImageClassifier(visual,head,normalize_features=True)
    with torch.no_grad():model.classifier.linear.bias.fill_(3.25)
    return model

def metrics_fixture(arm="L35"):
    route="S" if arm=="S" else "L35_GRAPH"
    last=dict(epoch=20,training_scope='full_train_fixed_epochs',config_sha256='a'*64,mode='full_visual',
        best_state=None,initial_validation=None,visual_input_resolution=336,position_interpolation='bicubic_7x7_to_10x10',
        history=[{'epoch':i,'train_loss':1.} for i in range(1,21)],optimizer={},scheduler={},scaler={},
        python_rng_state=(),numpy_rng_state=(),torch_rng_state=torch.tensor([1]),cuda_rng_state_all=[],confidence_weighting={})
    provenance=dict(competition_stage='stage2',clip_weight_sha256=fc.CLIP_SHA,train_manifest_sha256=fc.MANIFEST_SHA,class_mapping_sha256=fc.MAPPING_SHA)
    last.update(provenance);last.update(fresh_route_full_source_manifest_sha256=fc.FULL_SOURCE_SHA,fresh_route_full_arm=arm,selected_route=route,classifier_head_source=('legal_full_H1_frozen' if arm=='S' else 'joint_fixed_last_state'),classifier_head_restored_equal=(arm=='S'))
    final={k:last[k] for k in ('training_scope','config_sha256','mode','visual_input_resolution','position_interpolation')}
    final.update(provenance);final.update(selection_rule='fixed final epoch 20; no validation selection',normalize_features=True,head_initializer_sha256='h'*64)
    metrics=dict(training_scope='full_train_fixed_epochs',train_count=148695,validation_count=0,early_stop=None,initial_validation=None,best_top1=None,
        selected_checkpoint='final.pt',selected_checkpoint_sha256='f'*64,final_checkpoint_sha256='f'*64,last_checkpoint_sha256='l'*64,
        selection_rule=final['selection_rule'],config_sha256='a'*64,head_initializer={'sha256':'h'*64},
        visual_input_resolution=336,position_interpolation=last['position_interpolation'],history=copy.deepcopy(last['history']),trainable_total_parameters=88233966)
    metrics['trainable_total_parameters']=87849216 if arm=='S' else 88233966
    metrics['base_loss']={'kind':'sce','alpha':.5 if arm=='S' else .1,'beta':1.,'log_zero':-4.}
    for v in (metrics,final):
        for key in ('fresh_route_full_source_manifest_sha256','fresh_route_full_arm','selected_route','classifier_head_source','classifier_head_restored_equal'):v[key]=last[key]
    return metrics,last,final

class FinalTests(unittest.TestCase):
    def test_owned_cleanup_survives_proc_permission_failure_without_foreign_group(self):
        import run_final
        child=Mock(pid=991);child.poll.return_value=None
        with patch.object(run_final.os,'getpgid',return_value=991,create=True),patch.object(run_final,'process_start',side_effect=PermissionError('fixture')),patch.object(run_final.os,'killpg',create=True) as kill:
            run_final.stop_owned_child(child,'stable');kill.assert_called_once_with(991,run_final.signal.SIGTERM);child.wait.assert_called_once_with(timeout=20)
        child.reset_mock();child.poll.return_value=None
        with patch.object(run_final.os,'getpgid',return_value=992,create=True),patch.object(run_final.os,'killpg',create=True) as kill:
            with self.assertRaises(ValueError):run_final.stop_owned_child(child,'stable')
            kill.assert_not_called()

    def test_joint_restore_overrides_h1(self):
        original=toy();state=trainable_state_dict(original)
        initial=LinearClassifierHead(512,750)
        with torch.no_grad():initial.linear.bias.fill_(-8)
        restored,delta=fc.restore_joint(nn.Linear(3,512),initial,state,torch,expected_visual=2)
        self.assertTrue(torch.equal(restored.classifier.linear.bias,state['classifier.linear.bias']))
        self.assertEqual(delta['classifier.linear.bias'],11.25)

    def test_partial_head_rejected(self):
        state=trainable_state_dict(toy());state.pop('classifier.linear.bias')
        with self.assertRaises(ValueError):fc.restore_joint(nn.Linear(3,512),LinearClassifierHead(512,750),state,torch,expected_visual=2)

    def test_final_last_identity_and_corruption(self):
        state=trainable_state_dict(toy());last={'trainable_state':state};final=copy.deepcopy(last)
        fc.validate_joint_tensors(last,final,torch,expected_visual=2,expected_numel=sum(v.numel() for v in state.values()))
        final['trainable_state']['classifier.linear.bias'][0]+=1
        with self.assertRaises(ValueError):fc.validate_joint_tensors(last,final,torch,expected_visual=2,expected_numel=sum(v.numel() for v in state.values()))

    def test_full20_metrics_reject_partial_and_hash_drift(self):
        m,l,f=metrics_fixture()
        kw=dict(config_sha='a'*64,final_sha='f'*64,last_sha='l'*64,head_sha='h'*64,arm='L35')
        fc.validate_metrics(m,l,f,**kw)
        l['epoch']=19
        with self.assertRaises(ValueError):fc.validate_metrics(m,l,f,**kw)
        l['epoch']=20;m['final_checkpoint_sha256']='x'*64
        with self.assertRaises(ValueError):fc.validate_metrics(m,l,f,**kw)

    def test_full20_reject_validation_or_old152(self):
        for change in ('validation','old152'):
            m,l,f=metrics_fixture()
            if change=='validation':m['validation_count']=29737
            else:m['trainable_total_parameters']=87849216
            with self.assertRaises(ValueError):fc.validate_metrics(m,l,f,config_sha='a'*64,final_sha='f'*64,last_sha='l'*64,head_sha='h'*64,arm='L35')

    def test_real_adam_groups_scheduler_and_missing_moment(self):
        model=toy();visual=list(model.visual.parameters());head=list(model.classifier.parameters())
        opt=torch.optim.AdamW([dict(params=visual,lr=3.5e-6),dict(params=head,lr=5e-6)],weight_decay=.01)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
        for _ in range(20):
            for param in model.parameters():param.grad=torch.ones_like(param)
            opt.step();sched.step();opt.zero_grad()
        scaler=torch.amp.GradScaler('cpu')
        last={'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict()}
        fc.restore_optimizer(model,last,torch,5e-6)
        broken=copy.deepcopy(last);broken['optimizer']['state'].pop(next(iter(broken['optimizer']['state'])))
        with self.assertRaises(ValueError):fc.restore_optimizer(model,broken,torch,5e-6)

    def test_feature_hook_zero_extra_forward_and_same_logits(self):
        torch.manual_seed(15);model=toy().eval();x=torch.randn(3,3);scaled=torch.randn(3,3)
        # Give exact flips a real spatial dimension while retaining visual3 inputs.
        class Visual(nn.Module):
            def __init__(self,linear):super().__init__();self.linear=linear;self.calls=0
            def forward(self,v):self.calls+=1;return self.linear(v.flatten(1))
        model.visual=Visual(model.visual);x=x.reshape(3,1,1,3);scaled=scaled.reshape(3,1,1,3)
        with torch.inference_mode():baseline=fixed_four_view_logits(model,x,scaled)[2]
        model.visual.calls=0;hook=FourViewFeatureCollector(model.visual)
        with torch.inference_mode():actual=fixed_four_view_logits(model,x,scaled)[2]
        features=hook.finish(3);hook.close()
        self.assertTrue(torch.equal(baseline,actual));self.assertEqual(model.visual.calls,4);self.assertEqual(hook.calls,4)
        self.assertEqual(features.dtype,torch.float32);self.assertEqual(features.shape,(3,512))

    def test_postprocess_fixed_recipe_deterministic(self):
        logits=torch.tensor([[8.,0.,1.],[7.,0.,2.],[6.,1.,0.],[0.,5.,1.]])
        bias,a=fit_uniform_soft_bias(logits,temperature_multiplier=.5,iterations=60)
        again,b=fit_uniform_soft_bias(logits,temperature_multiplier=.5,iterations=60)
        self.assertTrue(torch.equal(bias,again));self.assertEqual(a,b)
        self.assertTrue(torch.isfinite(logits+.75*bias).all())

    def test_official_test_labels_path_duplicate_rejected(self):
        row=dict(sample_id='x',relative_path='1.jpg',source_split='test',internal_class_index='',official_class_id='')
        fc.validate_unlabeled_test_rows([row],expected_count=1)
        for mutate in (dict(internal_class_index='1'),dict(relative_path='../x.jpg'),dict(source_split='train')):
            bad=row|mutate
            with self.assertRaises(ValueError):fc.validate_unlabeled_test_rows([bad],expected_count=1)
        with self.assertRaises(ValueError):fc.validate_unlabeled_test_rows([row,row],expected_count=2)

    def test_dataset_only_pinned_override_allowed(self):
        original={'dataset':{'dataset_root':'old','allow_truncated_images':False,'sources':{'test':{'root':'__unavailable_test__','label_source':'none'}},'transforms':{'resize_size':336}}}
        overlay=copy.deepcopy(original);overlay['dataset']['dataset_root']=fc.OFFICIAL_DATA_ROOT;overlay['dataset']['sources']['test']['root']='test'
        fc.validate_dataset_override(original,overlay)
        overlay['dataset']['transforms']['resize_size']=384
        with self.assertRaises(ValueError):fc.validate_dataset_override(original,overlay)

    def test_csv_zip_exact_order_labels_and_no_overwrite(self):
        mapping=ClassMapping([ClassEntry('0001',0),ClassEntry('0099',1)])
        spec=SubmissionSpec(include_header=False,filename_mode='basename',official_class_id_pattern=r'\d{4}')
        with fixture_directory() as temp:
            output=Path(temp);names=['a.jpg','b.jpg']
            meta={k:'fixture' for k in ('run_id','checkpoint_sha256','config_hash','data_hash','git_commit','inference_command')}
            package=build_submission_package({'a.jpg':1,'b.jpg':0},names,mapping,output,run_metadata=meta,spec=spec)
            self.assertEqual(package.csv_path.read_text(),'a.jpg,0099\nb.jpg,0001\n')
            validate_submission_zip(package.zip_path,names,['0001','0099'],spec=spec)
            with self.assertRaises(FileExistsError):build_submission_package({'a.jpg':1,'b.jpg':0},names,mapping,output,run_metadata=meta,spec=spec)
            package.csv_path.write_text('b.jpg,0001\na.jpg,0099\n')
            with self.assertRaises(ValueError):validate_submission_csv(package.csv_path,names,['0001','0099'],spec=spec)
            with zipfile.ZipFile(package.zip_path,'w') as archive:archive.writestr('nested/pred_results.csv','a.jpg,0099\nb.jpg,0001\n')
            with self.assertRaises(ValueError):validate_submission_zip(package.zip_path,names,['0001','0099'],spec=spec)

    def test_download_inventory_byte_corruption(self):
        with fixture_directory() as temp:
            folder=Path(temp);(folder/'full').mkdir();(folder/'full/final.pt').write_bytes(b'f');(folder/'full/last.pt').write_bytes(b'l')
            files={n:{'path':'/remote/'+n,'sha256':fc.sha(folder/n),'bytes':1} for n in ('full/final.pt','full/last.pt')}
            manifest=dict(schema='aictrack1.fresh_route.final_return_manifest.v1',complete=True,files=files,full_training_source_sha256=fc.FULL_SOURCE_SHA,
                test_images_included=False,final_sha256=files['full/final.pt']['sha256'],last_sha256=files['full/last.pt']['sha256'])
            verify_return_files(folder,manifest);(folder/'full/final.pt').write_bytes(b'z')
            with self.assertRaises(ValueError):verify_return_files(folder,manifest)



    def test_fixed_graph_selected_point_margin_isolation_and_original_core(self):
        from fixed_final_graph import apply_fixed_graph,PARAMETERS
        from p3_graph import blocked_topk,mutual_graph,smooth,p7
        generator=np.random.default_rng(5);scores=generator.normal(size=(8,750)).astype(np.float32);scores[0]=-2;scores[0,0]=10
        features=generator.uniform(.5,1.,size=(8,512)).astype(np.float32);features/=np.linalg.norm(features,axis=1,keepdims=True);features[1]=0
        preds,transformed,detail=apply_fixed_graph(scores,features,deadline=time.monotonic()+30)
        self.assertTrue(np.array_equal(transformed[0],scores[0]));self.assertTrue(np.array_equal(transformed[1],scores[1]));self.assertEqual(detail['smoothing']['protected_raw_prediction_changes'],0)
        neighbors,weights,_=blocked_topk(features,20);transition,_,degree=mutual_graph(neighbors,weights,10);expected,_=smooth(scores,transition,degree,.3);reference,_=p7(expected)
        self.assertTrue(np.array_equal(transformed,expected));self.assertTrue(np.array_equal(preds,reference));self.assertFalse(detail['parameter_search']);self.assertEqual(detail['additional_model_forwards'],0)
        with self.assertRaises(ValueError):apply_fixed_graph(scores,features,deadline=time.monotonic()+30,parameters=PARAMETERS|{'lambda':.1})
        with self.assertRaises(Exception):apply_fixed_graph(scores,features,deadline=time.monotonic()-1)
        with self.assertRaises(ValueError):apply_fixed_graph(scores,features[:-1],deadline=time.monotonic()+30)

    def test_unbound_template_cannot_be_used_as_actual_source(self):
        with fixture_directory() as temp:
            root=Path(temp);(root/'source_manifest.json').write_text(json.dumps({'schema':'aictrack1.fresh_route.final_inference.template.v1','files':{}}))
            with self.assertRaises(ValueError):fc.verify_source(root)

    def test_real_entry_cli_imports_no_gpu_no_data(self):
        for name in ('infer_and_build_submission.py','audit_full_checkpoint.py','run_final.py','accept_return.py','prepare_final_source.py'):
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/name),'--help'],capture_output=True,text=True,env=os.environ.copy())
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('usage:',result.stdout)

if __name__=='__main__':unittest.main(verbosity=2)

"""Meaningful CPU-only FULL20 provenance, gradients, recovery, disk and queue checks."""
import argparse,copy,json,random,shutil,sys,unittest,uuid
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from torch import nn
from unittest.mock import patch
from p1_contract import ARMS,assets,validate_assets,validate_config,validate_selection,sha,verify_files,FULL_H1_SHA,MAPPING_SHA,_paired_identity_sha
from aictrack1.models.heads import LinearClassifier
from aictrack1.models.peft import PEFTImageClassifier
from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict,configure_classifier_trainability_from_state
from aictrack1.methods.confidence import ClasswiseHistoryWeighting
from aictrack1.engine.training_queue import QueueItem,run_serial_queue,read_training_lock
from train_n1_peft import _base_loss_per_sample,_weighted_base_loss
FULL_H1=None

@contextmanager
def fixture_directory():
    path=ROOT/'artifacts'/('cpu-'+uuid.uuid4().hex);path.mkdir(parents=True)
    try:yield path
    finally:
        if not path.resolve().is_relative_to((ROOT/'artifacts').resolve()):raise RuntimeError('fixture path escape')
        shutil.rmtree(path)

def selection_fixture():
    return dict(schema='aictrack1.p1.full_selection.v1',status='selected_for_full_training',selected_arm='J5',head_learning_rate=5e-6,
        checkpoint_role='last',checkpoint_epoch=20,training_scope='fixed_split_validation',train_count=118958,validation_count=29737,
        training_complete=True,decision_basis='current_stage_train_derived_validation',used_test_or_online_scores=False,
        training_source_manifest_sha256='1'*64,diagnostic_source_manifest_sha256='2'*64,selection_protocol_sha256='3'*64,
        confirmation_receipt=dict(path='/fixture/fresh_confirmation.json',sha256='5'*64),confirmation_receipt_sha256='5'*64,
        selection_reason='UNIT TEST FIXTURE ONLY',original_initial_free_bytes=13*1024**3,
        evidence={name:dict(path='/fixture/'+name,sha256='1'*64 if name=='training_source' else '4'*64)
            for name in ('checkpoint','config','identity','paired_report','training_source')})

class Tests(unittest.TestCase):
    def test_full_assets_and_single_selected_lr(self):
        import yaml
        payload=assets();validate_assets(payload)
        normalized=[]
        for arm in ARMS:
            raw=yaml.safe_load((ROOT/f'configs/experiment/p1_full_{arm}.yaml').read_text())['experiment']
            validate_config(raw,arm,payload)
            raw=copy.deepcopy(raw)
            for key in ('run_id','output_dir'):raw.pop(key)
            raw['training'].pop('head_learning_rate');normalized.append(raw)
        self.assertEqual(normalized[0],normalized[1]);self.assertEqual(normalized[0],normalized[2])
        self.assertEqual(verify_files()['status'],'ready_requires_positive_selection')
        for mutation in [dict(train_count=118958,validation_count=29737),dict(h1_training_scope='fold_train_fixed_final_epochs')]:
            bad=copy.deepcopy(payload);bad.update(mutation)
            with self.assertRaises(ValueError):validate_assets(bad)
        bad=copy.deepcopy(payload);bad['assets']['h1']['sha256']='45266a98329a59f708b12f394775640e032e50be8c44077a1037048f28c88753'
        with self.assertRaises(ValueError):validate_assets(bad)
        choice=selection_fixture();validate_selection(choice)
        for mutation in [dict(status='awaiting_selection'),dict(checkpoint_epoch=8),dict(checkpoint_role='best'),
                dict(used_test_or_online_scores=True),dict(training_scope='full_train_fixed_epochs'),dict(training_complete=False),
                dict(status='selected_recipe_for_confirmation'),dict(selected_arm='C',head_learning_rate=0.)]:
            bad=copy.deepcopy(choice);bad.update(mutation)
            with self.assertRaises(ValueError):validate_selection(bad)
        bad=copy.deepcopy(choice);bad.pop('confirmation_receipt')
        with self.assertRaises(ValueError):validate_selection(bad)
        recipe=copy.deepcopy(choice);recipe['status']='selected_recipe_for_confirmation';recipe.pop('confirmation_receipt')
        recipe.pop('confirmation_receipt_sha256');validate_selection(recipe,require_confirmation=False)
        with self.assertRaises(ValueError):validate_selection(recipe)
        self.assertEqual(_paired_identity_sha(dict(provenance=dict(identity_sha256='a'*64))),'a'*64)
        with self.assertRaises(ValueError):_paired_identity_sha(dict(identity_sha256='a'*64,provenance=dict(identity_sha256='b'*64)))

    def test_independent_fresh_confirmation_cannot_be_replaced_by_old_fold(self):
        from full_confirmation import REQUIRED_EVIDENCE,FRESH_SPLIT_SHA,validate_confirmation
        evidence={name:dict(path='/fixture/'+name,sha256='7'*64) for name in REQUIRED_EVIDENCE}
        evidence['split']['sha256']=FRESH_SPLIT_SHA
        value=dict(schema='aictrack1.p1.fresh_confirmation.v1',status='positive_confirmed_for_full_training',selected_arm='J5',
            split_seed=20261009,split_sha256=FRESH_SPLIT_SHA,train_count=118958,validation_count=29737,
            train_query_overlap=0,old_query_overlap=0,used_test_or_online_scores=False,test_access=False,
            initialization=dict(clip_sha256=assets()['assets']['clip']['sha256'],h1_epoch=100,h1_scope='fold_train_fixed_final_epochs'),
            runtime=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',actual_torch_threads=4,equal_between_arms=True),
            decision_gate=dict(status='pass',primary_delta_pp=.2,natural_p7_delta_pp=.1),
            root_review=dict(status='pass',reviewer='root',reason='UNIT TEST ONLY'),evidence=evidence,
            arms={name:dict(epoch=20,checkpoint_role='last',checkpoint_sha256='7'*64,config_sha256='7'*64,
                identity_sha256='7'*64,checkpoint_audit_sha256='7'*64) for name in ('C','J5')},
            h1_sha256='7'*64,h1_receipt_sha256='7'*64,confirmation_training_source_manifest_sha256='7'*64,
            diagnostic_source_manifest_sha256='7'*64,selection_protocol_sha256='7'*64,frozen_postprocess_sha256='7'*64)
        validate_confirmation(value)
        for mutation in [dict(split_sha256='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'),
                dict(old_query_overlap=1),dict(used_test_or_online_scores=True),dict(status='engineering_pass_only'),
                dict(selected_arm='C'),dict(root_review=dict(status='pending',reviewer='root',reason='UNIT TEST')),
                dict(decision_gate=dict(status='pass',primary_delta_pp=0.,natural_p7_delta_pp=.1)),
                dict(decision_gate=dict(status='pass',primary_delta_pp=.2,natural_p7_delta_pp=float('nan'))),
                dict(runtime=dict(OMP_NUM_THREADS='64',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',actual_torch_threads=64,equal_between_arms=True))]:
            bad=copy.deepcopy(value);bad.update(mutation)
            with self.assertRaises(ValueError):validate_confirmation(bad)
        bad=copy.deepcopy(value);bad['arms']['C']['epoch']=8
        with self.assertRaises(ValueError):validate_confirmation(bad)
        bad=copy.deepcopy(value);bad['evidence'].pop('runtime_C')
        with self.assertRaises(ValueError):validate_confirmation(bad)
        diagnostic=copy.deepcopy(value);diagnostic['decision_gate']['natural_p7_delta_pp']=-.1
        validate_confirmation(diagnostic)  # natural prior mismatch is reported, not a new algorithm gate.

    def test_actual_full_h1_strict_loader(self):
        import mainline_h1_contract as module
        self.assertIsNotNone(FULL_H1,'supply --full-h1 real local checkpoint')
        payload=assets();payload['assets']['h1']['path']=str(FULL_H1)
        with patch.object(module,'assets',return_value=payload):
            head,state=module.load_new_fold_head(FULL_H1,feature_dim=512,num_classes=750,
                expected_clip_weight_sha256=payload['assets']['clip']['sha256'],expected_class_mapping_sha256=MAPPING_SHA,
                expected_split_sha256=payload['assets']['split']['sha256'],forbid_full_train_initializer=False)
            self.assertEqual(sha(FULL_H1),FULL_H1_SHA);self.assertEqual(sum(p.numel() for p in head.parameters()),384750)
            self.assertEqual(state['row_binding']['counts'],{'train':148695,'validation':0})
            self.assertEqual(state['epoch'],100);self.assertIsNone(state['best_model'])
            with self.assertRaises(ValueError):module.load_new_fold_head(FULL_H1,forbid_full_train_initializer=True)

    def test_joint_gradients_and_actual_disk_full_state_resume(self):
        torch.set_num_threads(4)
        for arm,lr in ARMS.items():
            torch.manual_seed(7);random.seed(7);np.random.seed(7)
            model=PEFTImageClassifier(nn.Linear(5,8),LinearClassifier(8,4),normalize_features=True)
            for p in model.classifier.parameters():p.requires_grad_(lr>0)
            groups=[dict(params=list(model.visual.parameters()),lr=1.75e-6)]
            if lr:groups.append(dict(params=list(model.classifier.parameters()),lr=lr))
            opt=torch.optim.AdamW(groups,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            ids=torch.arange(8);targets=ids%4;ac=ClasswiseHistoryWeighting(ids,targets,momentum=.8,history_mix=.5,min_weight=.02,magnitude_mix=.75)
            def step():
                images=torch.randn(8,5)*(1+random.random()*.01+float(np.random.rand())*.01)
                opt.zero_grad(set_to_none=True);logits=model(images)
                loss=_weighted_base_loss(_base_loss_per_sample(logits,targets,kind='sce',q=.7,label_smoothing=0.),ac.weights_for(ids))
                scaler.scale(loss).backward();scaler.unscale_(opt)
                for p in model.classifier.parameters():
                    if lr:self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.norm()),0.)
                    else:self.assertIsNone(p.grad)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                scaler.step(opt);scaler.update();sched.step()
                ac.update_epoch(ids,torch.nn.functional.cross_entropy(logits.detach(),targets,reduction='none'),
                    logits.detach().softmax(1)[ids,targets],logits.detach().argmax(1))
                return float(loss.detach())
            original={name:p.detach().clone() for name,p in model.classifier.named_parameters()};step()
            rng=np.random.get_state();rng=(rng[0],rng[1].tolist(),*rng[2:])
            saved=dict(trainable_state=trainable_state_dict(model),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),
                confidence_weighting=ac.state_dict(),python_rng_state=random.getstate(),numpy_rng_state=rng,torch_rng_state=torch.random.get_rng_state(),
                training_scope='full_train_fixed_epochs',epoch=1,best_state=None,initial_validation=None,config_sha256='fixture')
            with fixture_directory() as folder:
                path=folder/'last.pt';torch.save(saved,path);del saved
                expected_loss=step();expected=trainable_state_dict(model);expected_ac=ac.state_dict();expected_opt=copy.deepcopy(opt.state_dict())
                state=torch.load(path,map_location='cpu',weights_only=True)
                self.assertEqual(state['training_scope'],'full_train_fixed_epochs');self.assertIsNone(state['best_state']);self.assertIsNone(state['initial_validation'])
                load_trainable_state_dict(model,state['trainable_state']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
                scaler.load_state_dict(state['scaler']);ac.load_state_dict(state['confidence_weighting'])
                random.setstate(state['python_rng_state']);np.random.set_state(state['numpy_rng_state']);torch.random.set_rng_state(state['torch_rng_state'])
                self.assertEqual(step(),expected_loss)
                for key,value in trainable_state_dict(model).items():self.assertTrue(torch.equal(value,expected[key]),key)
                for index,values in opt.state_dict()['state'].items():
                    for key,value in values.items():self.assertTrue(torch.equal(value,expected_opt['state'][index][key]))
                for key,value in ac.state_dict().items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,expected_ac[key]),key)
                    else:self.assertEqual(value,expected_ac[key])
                self.assertEqual(sched.last_epoch,2);self.assertEqual(sched.T_max,20)
            changed=any(not torch.equal(p.detach(),original[k]) for k,p in model.classifier.named_parameters());self.assertEqual(changed,lr>0)
            self.assertEqual(configure_classifier_trainability_from_state(model.classifier,expected),lr>0);load_trainable_state_dict(model,expected)

    def test_atomic_save_reserve_and_original_disk_baseline(self):
        import mainline_guard as guard
        from types import SimpleNamespace
        array=torch.arange(16);self.assertEqual(guard.tensor_storage_bytes({'a':array,'alias':array}),array.untyped_storage().nbytes())
        env=dict(P1_DEADLINE='9999999999',P1_INITIAL_FREE=str(12*1024**3),P1_DISK_MAX=str(20*1024**3),P1_MIN_FREE=str(3*1024**3))
        with patch.dict('os.environ',env),patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=4*1024**3)):
            guard._resource_check(0)
            with self.assertRaises(RuntimeError):guard._resource_check(2*1024**3)
        env['P1_DISK_MAX']=str(2*1024**3)
        with patch.dict('os.environ',env),patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=8*1024**3)):
            with self.assertRaises(RuntimeError):guard._resource_check(0)

    def test_original_serial_queue_exact_last_resume(self):
        with fixture_directory() as root:
            (root/'scripts').mkdir();(root/'scripts/train_n1_peft.py').write_text('# fixture\n')
            out=root/'artifacts/experiments/full';out.mkdir(parents=True);(out/'last.pt').write_bytes(b'fixture')
            lock=root/'canonical.lock.json';seen=[]
            def launch(command,cwd,handle,on_start):
                self.assertIsNotNone(read_training_lock(lock));self.assertEqual(command[-2],'--resume');self.assertEqual(Path(command[-1]),out/'last.pt')
                on_start(1234);self.assertEqual(read_training_lock(lock).trainer_pid,1234);seen.append(command)
                for name in ['config.yaml','command.txt','final.pt']:(out/name).write_text('fixture')
                (out/'metrics.json').write_text(json.dumps(dict(selected_checkpoint='final.pt')));return 0
            item=QueueItem('full','config.yaml','artifacts/experiments/full','CPU fixture','scripts/train_n1_peft.py','artifacts/experiments/full/last.pt')
            self.assertEqual(run_serial_queue(root=root,python=Path(sys.executable),logs=root/'logs',queue_name='full-cpu-fixture',
                items=[item],process_probe=lambda:[],launcher=launch,lock_path=lock),0)
            self.assertEqual(len(seen),1);self.assertFalse(lock.exists())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full-h1',type=Path,required=True);args=parser.parse_args();FULL_H1=args.full_h1
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt=dict(schema='aictrack1.p1.full_cpu_smoke.v1',status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),python=sys.executable,torch=torch.__version__,cuda_used=False,
        real_full_h1_sha256=sha(FULL_H1),toy_resume_not_real_clip_throughput=True)
    (ROOT/'cpu_smoke_result.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(0 if result.wasSuccessful() else 1)

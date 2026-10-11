"""CPU-only tests of the actual loss/state/queue interfaces used by P1."""
import copy,json,random,sys,unittest,uuid,shutil
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from torch import nn
from aictrack1.models.heads import LinearClassifier
from aictrack1.models.peft import PEFTImageClassifier
from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict,configure_classifier_trainability_from_state
from aictrack1.methods.confidence import ClasswiseHistoryWeighting
from aictrack1.methods.confidence_observation import observation_protocol,validate_observation_resume
from aictrack1.engine.training_queue import QueueItem,run_serial_queue,read_training_lock
from train_n1_peft import _base_loss_per_sample,_weighted_base_loss
from p1_contract import ARMS,assets,validate_assets,validate_config,sha,verify_files

@contextmanager
def fixture_directory(prefix):
    path=ROOT/'artifacts'/(prefix+uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:yield str(path)
    finally:
        if not path.resolve().is_relative_to((ROOT/'artifacts').resolve()):raise RuntimeError('fixture escape')
        shutil.rmtree(path)

class Tests(unittest.TestCase):
    def test_joint_gradients_and_exact_disk_resume(self):
        torch.set_num_threads(1)
        for arm,lr in ARMS.items():
            torch.manual_seed(3);random.seed(3);np.random.seed(3)
            model=PEFTImageClassifier(nn.Linear(5,8),LinearClassifier(8,4),normalize_features=True)
            for p in model.classifier.parameters():p.requires_grad_(lr>0)
            groups=[dict(params=list(model.visual.parameters()),lr=1.75e-6)]
            if lr:groups.append(dict(params=list(model.classifier.parameters()),lr=lr))
            opt=torch.optim.AdamW(groups,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            ids=torch.arange(8);targets=torch.arange(8)%4
            confidence=ClasswiseHistoryWeighting(ids,targets,momentum=.8,history_mix=.5,min_weight=.02,magnitude_mix=.75)
            protocol=observation_protocol('strong',20261008)
            def step():
                x=torch.randn(8,5)*(1.+random.random()*.01+float(np.random.rand())*.01)
                opt.zero_grad(set_to_none=True);logits=model(x)
                loss=_weighted_base_loss(_base_loss_per_sample(logits,targets,kind='sce',q=.7,label_smoothing=0.,sce_alpha=.5),confidence.weights_for(ids))
                scaler.scale(loss).backward();scaler.unscale_(opt)
                for p in model.classifier.parameters():
                    if lr:self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.norm()),0.)
                    else:self.assertIsNone(p.grad)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                scaler.step(opt);scaler.update();sched.step()
                ce=torch.nn.functional.cross_entropy(logits.detach(),targets,reduction='none')
                confidence.update_epoch(ids,ce,logits.detach().softmax(1)[ids,targets],logits.detach().argmax(1))
                return float(loss.detach())
            original_head={k:v.detach().clone() for k,v in model.classifier.named_parameters()}
            step();rng=np.random.get_state();rng=(rng[0],rng[1].tolist(),*rng[2:])
            saved=dict(trainable_state=trainable_state_dict(model),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                scaler=scaler.state_dict(),python_rng_state=random.getstate(),numpy_rng_state=rng,torch_rng_state=torch.random.get_rng_state(),
                confidence_weighting=confidence.state_dict(),confidence_observation_protocol=protocol,config_sha256='cpu-test',epoch=1)
            (ROOT/'artifacts').mkdir(exist_ok=True)
            with fixture_directory('p1-resume-') as tmp:
                path=Path(tmp)/'last.pt';torch.save(saved,path);del saved
                expected_loss=step();expected=trainable_state_dict(model);expected_conf=confidence.state_dict()
                expected_opt=copy.deepcopy(opt.state_dict());expected_scheduler=copy.deepcopy(sched.state_dict())
                state=torch.load(path,map_location='cpu',weights_only=True)
                validate_observation_resume(state,protocol)
                self.assertEqual(state['epoch'],1);self.assertEqual(state['config_sha256'],'cpu-test')
                load_trainable_state_dict(model,state['trainable_state']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
                scaler.load_state_dict(state['scaler']);confidence.load_state_dict(state['confidence_weighting'])
                random.setstate(state['python_rng_state']);np.random.set_state(state['numpy_rng_state']);torch.random.set_rng_state(state['torch_rng_state'])
                self.assertEqual(step(),expected_loss)
                for name,value in trainable_state_dict(model).items():self.assertTrue(torch.equal(value,expected[name]),name)
                for name,value in confidence.state_dict().items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,expected_conf[name]),name)
                    else:self.assertEqual(value,expected_conf[name])
                self.assertEqual(sched.state_dict(),expected_scheduler)
                for index,values in opt.state_dict()['state'].items():
                    for name,value in values.items():self.assertTrue(torch.equal(value,expected_opt['state'][index][name]))
            head_keys={k for k in expected if k.startswith('classifier.')}
            self.assertEqual(head_keys,{'classifier.linear.weight','classifier.linear.bias'} if lr else set())
            changed=any(not torch.equal(p.detach(),original_head[k]) for k,p in model.classifier.named_parameters())
            self.assertEqual(changed,lr>0)
            for p in model.classifier.parameters():p.requires_grad_(False)
            self.assertEqual(configure_classifier_trainability_from_state(model.classifier,expected),lr>0)
            load_trainable_state_dict(model,expected)
            bad=dict(expected);bad['classifier.unknown']=torch.tensor(0.)
            with self.assertRaises(ValueError):load_trainable_state_dict(model,bad)

    def test_bindings_and_config_single_variable(self):
        import yaml
        payload=assets();validate_assets(payload)
        configs={arm:yaml.safe_load((ROOT/f'configs/experiment/p1_{arm}.yaml').read_text())['experiment'] for arm in ARMS}
        for arm,raw in configs.items():validate_config(raw,arm,payload)
        parent=json.loads((ROOT/'inputs/parent_C_config.json').read_text(encoding='utf-8'))
        candidate=copy.deepcopy(configs['S'])
        for key in ('run_id','output_dir'):candidate[key]=parent[key]
        candidate['base_loss']['alpha']=.1
        self.assertEqual(candidate,parent)
        for mutation in (dict(status='unbound'),dict(h1_training_scope='full_train_fixed_epochs'),dict(h1_validation_used_for_selection=True)):
            bad=dict(payload);bad.update(mutation)
            with self.assertRaises(ValueError):validate_assets(bad)
        self.assertEqual(verify_files()['status'],'bound')

    def test_canonical_queue_explicit_n1_resume(self):
        # Real atomic lease implementation + injected CPU launcher: no native process/GPU probe.
        (ROOT/'artifacts').mkdir(exist_ok=True)
        with fixture_directory('p1-queue-') as tmp:
            root=Path(tmp);(root/'scripts').mkdir();(root/'scripts/train_n1_peft.py').write_text('# fixture\n')
            out=root/'artifacts/experiments/J5';out.mkdir(parents=True);(out/'last.pt').write_bytes(b'toy-checkpoint')
            lock=root/'canonical.lock.json';seen=[]
            def launcher(command,cwd,handle,on_start):
                record=read_training_lock(lock);self.assertIsNotNone(record);self.assertEqual(record.config,'config.yaml')
                self.assertEqual(command[-2],'--resume');self.assertEqual(Path(command[-1]),out/'last.pt')
                on_start(12345);self.assertEqual(read_training_lock(lock).trainer_pid,12345);seen.append(command)
                (out/'config.yaml').write_text('fixture');(out/'command.txt').write_text('fixture');(out/'final.pt').write_bytes(b'fixture')
                (out/'metrics.json').write_text(json.dumps(dict(selected_checkpoint='final.pt')));return 0
            item=QueueItem('J5','config.yaml','artifacts/experiments/J5','resume fixture','scripts/train_n1_peft.py','artifacts/experiments/J5/last.pt')
            self.assertEqual(run_serial_queue(root=root,python=Path(sys.executable),logs=root/'logs',queue_name='cpu-resume-fixture',items=[item],
                process_probe=lambda:[],launcher=launcher,lock_path=lock),0)
            self.assertEqual(len(seen),1);self.assertFalse(lock.exists())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt=dict(schema='aictrack1.supervision.cpu_smoke.v1',status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),python=sys.executable,torch=torch.__version__,cuda_used=False,
        scope='actual loss/state/AC/RNG/optimizer/scheduler/scaler and queue interfaces; toy model, not CLIP throughput')
    (ROOT/'cpu_smoke_result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)

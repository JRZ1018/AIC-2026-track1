"""Actual CPU loss/AC/RNG/AdamW/Cosine20/scaler/head disk-resume science fixtures."""
import copy,json,math,random,shutil,sys,unittest,uuid
from pathlib import Path
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from torch import nn
import full_route_contract as c
import full_route_runtime as rt
from aictrack1.models.heads import LinearClassifier
from aictrack1.models.peft import PEFTImageClassifier
from aictrack1.engine.peft import trainable_state_dict,load_trainable_state_dict,configure_classifier_trainability_from_state
from aictrack1.methods.confidence import ClasswiseHistoryWeighting
from aictrack1.methods.confidence_observation import observation_protocol,validate_observation_resume
from train_n1_peft import _base_loss_per_sample,_weighted_base_loss
@contextmanager
def fixture():
    path=ROOT/'artifacts'/('cpu_resume_'+uuid.uuid4().hex);path.mkdir(parents=True)
    try:yield path
    finally:
        if not path.resolve().is_relative_to((ROOT/'artifacts').resolve()):raise ValueError('fixture escape')
        shutil.rmtree(path)
class RecoveryTests(unittest.TestCase):
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
            saved=dict(trainable_state=trainable_state_dict(model),optimizer=opt.state_dict(),scheduler=sched.state_dict(),scaler=scaler.state_dict(),confidence_weighting=confidence.state_dict(),confidence_observation_protocol=protocol,python_rng_state=random.getstate(),numpy_rng_state=rng,torch_rng_state=torch.random.get_rng_state(),epoch=1,training_scope='full_train_fixed_epochs',best_state=None,initial_validation=None,selection_rule='fixed_final')
            with fixture() as tmp:
                torch.save(saved,tmp/'last.pt');del saved;expected_loss=step();expected=trainable_state_dict(model);expected_conf=confidence.state_dict();expected_opt=copy.deepcopy(opt.state_dict());expected_sched=copy.deepcopy(sched.state_dict())
                state=torch.load(tmp/'last.pt',weights_only=True,map_location='cpu');self.assertEqual(state['training_scope'],'full_train_fixed_epochs');self.assertIsNone(state['best_state']);self.assertIsNone(state['initial_validation']);validate_observation_resume(state,protocol);load_trainable_state_dict(model,state['trainable_state']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler']);confidence.load_state_dict(state['confidence_weighting']);random.setstate(state['python_rng_state']);np.random.set_state(state['numpy_rng_state']);torch.random.set_rng_state(state['torch_rng_state'])
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

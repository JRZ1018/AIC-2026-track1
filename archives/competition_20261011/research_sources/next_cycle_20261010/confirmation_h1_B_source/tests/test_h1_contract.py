"""CPU fixtures only; synthetic model states are never candidate H1 checkpoints."""
from __future__ import annotations
import copy,hashlib,io,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import confirmation_contract as c
import owned_launch as g
import torch,numpy as np
from aictrack1.engine.frozen_head import FrozenHeadTrainer,FrozenHeadConfig
from aictrack1.models.heads import build_classifier_head

def budget():return dict(schema='aictrack1.next_cycle10.confirmation_budget.v1',root_authorized=True,phase_start_unix=10,deadline_unix=20,original_initial_free_bytes=13013950464,capacity_added_bytes=32212254720,adjusted_initial_free_bytes=45226205184,actual_disk_total_bytes=128849018880,disk_max_bytes=20*1024**3,min_free_bytes=3*1024**3)
def admission():
 from B_admission_contract import SCIENTIFIC_TEMPLATE_SHA,MATH_SOURCE_SHA,MATH_PROTOCOL_SHA,PRIMARY_SCOPE
 return dict(schema='aictrack1.next_cycle10.B_admission.v1',root_authorized=True,complete=True,selected_arm='B25',soft_coefficient=.25,exploration_balanced_delta_pp=.1,exploration_paired_sha256='1'*64,root_risk_review_sha256='2'*64,exploration_training_source_sha256='5'*64,scientific_template_source_sha256=SCIENTIFIC_TEMPLATE_SHA,math_source_sha256=MATH_SOURCE_SHA,math_protocol_sha256=MATH_PROTOCOL_SHA,primary_scope=PRIMARY_SCOPE,query_definition_source_sha256=c.QUERY_SOURCE_SHA,new_H1_checkpoint_sha256=None,final_model='student_only',test_access=False)

class Contracts(unittest.TestCase):
    def test_unbound_rejected_for_training(self):
        with self.assertRaises(ValueError):c.verify_source()
        self.assertEqual(c.verify_source(False)['status'],'unbound_template')
    def test_exact_preserved_engine_queue(self):
        receipt=c.read_json(ROOT/'COPY_RECEIPT.json')
        self.assertEqual(c.sha(ROOT/'src/aictrack1/engine/training_queue.py'),receipt['original_canonical_queue_sha256'])
        self.assertEqual(c.sha(ROOT/'src/aictrack1/engine/frozen_head.py'),receipt['original_frozen_head_engine_sha256'])
        self.assertEqual(c.sha(ROOT/'scripts/owned_launch.py'),receipt['owned_launch_sha256'])
    def test_mapping_byte_SHA_distinct_from_logical_digest(self):
        from aictrack1.data.class_mapping import ClassMapping
        entries=[ClassMapping.from_csv]
        fixture=ROOT/'test_fixtures/classes.csv'
        value=c.audit_class_mapping(fixture)
        self.assertEqual(value['file_sha256'],'0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be')
        self.assertEqual(value['logical_mapping_sha256'],'811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879')
        self.assertNotEqual(value['file_sha256'],value['logical_mapping_sha256'])
    def test_mapping_modified_bytes_rejected(self):
        with tempfile.TemporaryDirectory(prefix='H1_classes_byte_fixture_') as td:
            p=Path(td)/'classes.csv';p.write_bytes((ROOT/'test_fixtures/classes.csv').read_bytes()+b'\n')
            with self.assertRaises(ValueError):c.audit_class_mapping(p)
    def test_metadata_positive(self):
        a=c.normalized_assets();self.assertEqual(a['h1_row_binding'],c.read_json(ROOT/'inputs/row_binding.json'));self.assertEqual(a['counts'],{'train':118958,'validation':29737})
    def test_budget_active(self):self.assertEqual(c.validate_budget(budget(),15)['deadline_unix'],20)
    def test_budget_deadline_rejected(self):
        with self.assertRaises(ValueError):c.validate_budget(budget(),20)
    def test_budget_baseline_reset_rejected(self):
        b=budget();b['adjusted_initial_free_bytes']+=1
        with self.assertRaises(ValueError):c.validate_budget(b)
    def test_admission_actual_positive(self):self.assertEqual(c.validate_admission(admission())['selected_arm'],'B25')
    def test_admission_nonpositive_rejected(self):
        a=admission();a['exploration_balanced_delta_pp']=0
        with self.assertRaises(ValueError):c.validate_admission(a)
    def test_admission_future_sha_rejected(self):
        a=admission();a['new_H1_checkpoint_sha256']='3'*64
        with self.assertRaises(ValueError):c.validate_admission(a)
    def test_admission_wrong_definition_rejected(self):
        a=admission();a['query_definition_source_sha256']='4'*64
        with self.assertRaises(ValueError):c.validate_admission(a)
    def test_B_admission_old_A_raw_nan_wrong_math_rejected(self):
        for k,v in [('selected_arm','A3'),('primary_scope','raw_P7'),('exploration_balanced_delta_pp',float('nan')),('scientific_template_source_sha256','7'*64),('math_source_sha256','8'*64),('soft_coefficient',.5),('test_access',True),('final_model','EMA')]:
            a=admission();a[k]=v
            with self.assertRaises(ValueError):c.validate_admission(a)
    def test_B50_positive_and_explicit30GiB_allowed(self):
        a=admission();a['selected_arm']='B50';a['soft_coefficient']=.5;self.assertEqual(c.validate_admission(a)['selected_arm'],'B50');b=budget();b['disk_max_bytes']=30*1024**3;self.assertEqual(c.validate_budget(b)['disk_max_bytes'],30*1024**3)
    def test_actual_CPU_trainer_disk_resume_nextstep(self):
        torch.set_num_threads(4);torch.manual_seed(20261008);np.random.seed(20261008)
        x=torch.randn(12,512);y=torch.arange(12)%750
        def make():return FrozenHeadTrainer(build_classifier_head('linear',512,750),FrozenHeadConfig(**c.HEAD_CONFIG),device='cpu',num_classes=750)
        first=make();first.train_epoch(x,y);state=first.state_dict()
        with tempfile.TemporaryDirectory(prefix='H1_synthetic_CPU_fixture_') as td:
            p=Path(td)/'synthetic.pt';torch.save(state,p);loaded=torch.load(p,map_location='cpu',weights_only=True)
            restored=make();restored.load_state_dict(loaded)
            self.assertEqual(restored.epoch,1);self.assertEqual(restored.scheduler.last_epoch,1);self.assertEqual(len(restored.optimizer.state),2)
            first.load_state_dict(copy.deepcopy(loaded));first.train_epoch(x,y);restored.load_state_dict(copy.deepcopy(loaded));restored.train_epoch(x,y)
            for key,value in first.model.state_dict().items():self.assertTrue(torch.equal(value,restored.model.state_dict()[key]))
            self.assertEqual(restored.epoch,2)
    def test_GO_order_and_fail_closed(self):
        events=[]
        class Pipe(io.BytesIO):
            def write(self,b):events.append('GO');return super().write(b)
        class Child:
            pid=777;stdin=Pipe()
            def poll(self):return None
        child=Child();owner=dict(pid=777,pgrp=777,session=777,start_ticks='1')
        g.start_with_go(['python','trainer'],ROOT,None,{},[],lambda pid:events.append('ATTACH'),lambda ch,own:events.append('READY'),lambda ch,own:events.append('CREATED'),lambda *a:events.append('HALT'),popen=lambda *a,**k:child,capture=lambda *a,**k:owner)
        self.assertLess(events.index('ATTACH'),events.index('GO'));self.assertLess(events.index('READY'),events.index('GO'));self.assertTrue(child.stdin.closed)
    def test_GO_failure_no_grant_cleanup(self):
        events=[]
        class Child:
            pid=888;stdin=io.BytesIO()
            def poll(self):return None
        child=Child();owner=dict(pid=888,pgrp=888,session=888,start_ticks='1')
        def fail(pid):raise RuntimeError('fixture attach failed')
        with self.assertRaises(RuntimeError):g.start_with_go(['python','trainer'],ROOT,None,{},[],fail,lambda *a:None,lambda *a:None,lambda *a:events.append('HALT'),popen=lambda *a,**k:child,capture=lambda *a,**k:owner,stop=lambda *a:events.append('CLEANUP'))
        self.assertTrue(child.stdin.closed);self.assertEqual(events,['CLEANUP'])
    def test_unknown_owner_rejected(self):
        with self.assertRaises(ValueError):g.own_identity(dict(pid=12,pgrp=13,session=12,start_ticks='1'),12,[])
if __name__=='__main__':unittest.main()

"""Necessary capacity/full-phase/env and actual retry provenance compatibility only."""
import copy,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import capacity_contract as k
import full_route_contract as c
import mainline_guard as g
from test_retry_chain_reference import RetryChainTests
from test_confirmation_reference import ReceiptEvidenceTests
class CapacityTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  # Actual fixture SHA/localpath are explicitly substitutes, never production ba819.
  self.path=self.root/'capacity.json'
  value=dict(schema='aictrack1.fresh_route.actual_disk_expansion_recovery.v1',root_authorized=True,original_data_total_bytes=k.OLD_TOTAL,actual_data_total_bytes=k.TOTAL,data_capacity_increase_bytes=k.CAPACITY,original_initial_free_bytes=k.INITIAL,adjusted_initial_free_bytes=k.ADJUSTED,original_shared_budget=k.SHARED_REF,prior_failure=k.FAILURE_REF,old_source_or_budget_modified=False,test_access=False,exact_prior_groups_closed=True,canonical_naturally_idle=True)
  self.path.write_text(json.dumps(value));self.ref=dict(path=str(self.path.resolve()),sha256=k.sha(self.path));p=patch.object(k,'RECOVERY_REF',self.ref);p.start();self.addCleanup(p.stop)
  self.now=time.time();self.b=dict(schema='aictrack1.fresh_route.full.budget.v1',root_authorized=True,phase_start_unix=self.now-1,deadline_unix=self.now+7200,initial_free_bytes=k.INITIAL,capacity_increase_bytes=k.CAPACITY,adjusted_initial_free_bytes=k.ADJUSTED,actual_data_total_bytes=k.TOTAL,disk_expansion_recovery=self.ref,disk_max_bytes=16*1024**3,min_free_bytes=3*1024**3)
 def test_real_capacity_ref_and_independent_new_full_phase(self):
  self.assertEqual(c.validate_budget(self.b,now=self.now),self.b)
  later=dict(self.b,phase_start_unix=1791540000.,deadline_unix=1791560000.)
  self.assertEqual(c.validate_budget(later,now=1791540001.),later)
  for bad in [{'capacity_increase_bytes':0},{'adjusted_initial_free_bytes':38303313920},{'initial_free_bytes':k.ADJUSTED},{'actual_data_total_bytes':k.TOTAL+1},{'disk_expansion_recovery':None}]:
   with self.subTest(bad=bad),self.assertRaises(ValueError):c.validate_budget(self.b|bad,now=self.now)
  self.path.write_text('{}')
  with self.assertRaises(ValueError):c.validate_budget(self.b,now=self.now)
 def test_original_consumption_and_atomic_floor(self):
  proof=k.resource_ok(self.b,38303313920,k.TOTAL,reserve=1024**3,now=self.now);self.assertEqual(proof['consumed_bytes'],6922891264)
  for free in (k.ADJUSTED-self.b['disk_max_bytes'],3*1024**3):
   with self.subTest(free=free),self.assertRaises(ValueError):k.resource_ok(self.b,free,k.TOTAL,reserve=1,now=self.now)
 def test_changed_capacity_and_full_clock_fail_closed(self):
  for total in (k.OLD_TOTAL,k.TOTAL+1):
   with self.subTest(total=total),self.assertRaises(ValueError):k.resource_ok(self.b,38303313920,total,now=self.now)
  for now in (self.b['phase_start_unix']-1,self.b['deadline_unix']):
   with self.subTest(now=now),self.assertRaises(ValueError):k.resource_ok(self.b,38303313920,k.TOTAL,now=now)
 def test_trainer_actual_guard_rejects_old_raw_environment(self):
  (self.root/'inputs').mkdir();(self.root/'inputs/full_budget.json').write_text(json.dumps(self.b))
  env=dict(P1_DEADLINE=str(self.b['deadline_unix']),P1_INITIAL_FREE=str(k.ADJUSTED),P1_DISK_MAX=str(self.b['disk_max_bytes']),P1_MIN_FREE=str(self.b['min_free_bytes']))
  with patch.object(g,'ROOT',self.root),patch.dict(os.environ,env),patch.object(g.shutil,'disk_usage',return_value=SimpleNamespace(free=38303313920,total=k.TOTAL)):
   g._resource_check(1024**3)
   with patch.dict(os.environ,{'P1_INITIAL_FREE':str(k.INITIAL)}),self.assertRaises(RuntimeError):g._resource_check(0)
   with patch.object(g.shutil,'disk_usage',return_value=SimpleNamespace(free=38303313920,total=k.TOTAL+1)),self.assertRaises(ValueError):g._resource_check(0)
 def test_missing_retry_refs_cannot_pass_base_writer(self):
  v=dict(schema='aictrack1.fresh_route.confirmation.v1',status='complete_positive_fresh_route_confirmation',selected_route='J5_GRAPH',selected_arm='J5',primary_scope='J5_graph_P7_vs_C',checkpoint_epoch=20,selection_rule='fixed_final20',training_scope='fixed_split_validation',split_sha256=c.FRESH_SPLIT_SHA,h1_sha256=c.FRESH_H1_SHA,source_h1_manifest_sha256=c.FRESH_H1_SOURCE_SHA,test_access=False,used_test_or_online_scores=False,root_risk_reviewed=True,primary_delta_vs_C_pp=.1,natural_delta_vs_C_pp=-.2,training_source_manifest_sha256='a'*64,diagnostic_source_manifest_sha256='b'*64,admission_receipt_sha256='c'*64)
  report=self.root/'report.json';report.write_text(json.dumps(dict(append_stage_context={'fixture':True})));v['evidence']={'paired_report':dict(path=str(report.resolve()),sha256=k.sha(report))}
  with self.assertRaisesRegex(ValueError,'nine extra'):c.verify_confirmation(v)
if __name__=='__main__':unittest.main(verbosity=2)

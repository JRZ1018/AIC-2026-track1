"""Metadata-only fixtures for one capacity addition and original cumulative ledger."""
from __future__ import annotations
import copy,json,sys,time,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import final_contract as fc
import final_capacity_contract as cap

class FinalCapacityTests(unittest.TestCase):
    def fixture(self):
        folder=ROOT/'artifacts'/'test_fixtures'/('capacity_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
        recovery=dict(schema='aictrack1.fresh_route.actual_disk_expansion_recovery.v1',root_authorized=True,
            original_data_total_bytes=cap.OLD_TOTAL,actual_data_total_bytes=cap.TOTAL,data_capacity_increase_bytes=cap.CAPACITY,
            original_initial_free_bytes=cap.INITIAL,adjusted_initial_free_bytes=cap.ADJUSTED,
            original_shared_budget=cap.SHARED_REF,prior_failure=cap.FAILURE_REF,old_source_or_budget_modified=False,
            test_access=False,exact_prior_groups_closed=True,canonical_naturally_idle=True)
        path=folder/'synthetic_recovery_metadata.json';path.write_text(json.dumps(recovery)+'\n',encoding='utf-8')
        ref=dict(path=str(path),sha256=cap.sha(path))
        budget=dict(initial_free_bytes=cap.INITIAL,capacity_increase_bytes=cap.CAPACITY,
            adjusted_initial_free_bytes=cap.ADJUSTED,actual_data_total_bytes=cap.TOTAL,disk_expansion_recovery=ref)
        policy=dict(deadline_unix=time.time()+60,initial_free_bytes=cap.INITIAL,disk_max_bytes=12*fc.GIB,
            min_free_bytes=3*fc.GIB,min_available_memory_bytes=2*fc.GIB)
        return budget,policy,ref

    def test_actual_policy_binding_retains_original_and_consumes_one_capacity_addition(self):
        budget,policy,ref=self.fixture();before=copy.deepcopy(budget)
        with patch.object(cap,'RECOVERY_REF',ref):
            actual=fc.bind_inference_capacity_policy(policy,budget,started_at_unix=time.time()-1)
            with patch.object(fc.shutil,'disk_usage',return_value=SimpleNamespace(total=cap.TOTAL,free=38303313920)):
                self.assertEqual(fc.resource_check(ROOT,actual,reserve=fc.GIB),38303313920)
            ledger=cap.resource_ok(actual,38303313920,cap.TOTAL,reserve=fc.GIB)
        self.assertEqual(ledger['consumed_bytes'],6922891264)
        self.assertEqual(actual['initial_free_bytes'],13013950464)
        self.assertEqual(actual['adjusted_initial_free_bytes'],45226205184)
        self.assertEqual(budget,before)

    def test_observed_free_double_capacity_or_missing_actual_recovery_rejected(self):
        budget,policy,ref=self.fixture()
        with patch.object(cap,'RECOVERY_REF',ref):
            for key,value in [('initial_free_bytes',38303313920),('adjusted_initial_free_bytes',cap.ADJUSTED+cap.CAPACITY),('capacity_increase_bytes',0),('disk_expansion_recovery',None)]:
                bad=copy.deepcopy(budget);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):fc.bind_inference_capacity_policy(policy,bad)
            with self.assertRaises(ValueError):fc.bind_inference_capacity_policy(policy|{'initial_free_bytes':38303313920},budget)
            path=Path(ref['path']);path.write_text(path.read_text().replace('"test_access": false','"test_access": true'),encoding='utf-8')
            with self.assertRaises(ValueError):fc.bind_inference_capacity_policy(policy,budget)

    def test_changed_total_atomic_growth_and_expired_final_deadline_rejected(self):
        budget,policy,ref=self.fixture()
        with patch.object(cap,'RECOVERY_REF',ref):
            actual=fc.bind_inference_capacity_policy(policy,budget)
            for total in (cap.TOTAL-1,cap.TOTAL+cap.CAPACITY):
                with patch.object(fc.shutil,'disk_usage',return_value=SimpleNamespace(total=total,free=38303313920)):
                    with self.assertRaises(ValueError):fc.resource_check(ROOT,actual)
            with patch.object(fc.shutil,'disk_usage',return_value=SimpleNamespace(total=cap.TOTAL,free=cap.ADJUSTED-actual['disk_max_bytes'])):
                fc.resource_check(ROOT,actual)
                with self.assertRaises(ValueError):fc.resource_check(ROOT,actual,reserve=1)
            with self.assertRaises(ValueError):fc.bind_inference_capacity_policy(policy|{'deadline_unix':time.time()-1},budget)
            with self.assertRaises(ValueError):cap.resource_ok(actual,38303313920,cap.TOTAL,now=actual['deadline_unix'])

    def test_actual_reviewed_full_manifest_and_all_provider_members_are_bound(self):
        import hashlib
        reference=ROOT/'references/UPSTREAM_FULL_TEMPLATE.json'
        self.assertEqual(hashlib.sha256(reference.read_bytes()).hexdigest(),'578eb15ce02421e16e3476b063c406fdf0322d8cc63233e740c29d9f51ab4d71')
        full=ROOT.parent/'fresh_route_full_expanded_training_source';manifest=json.loads(reference.read_text())
        self.assertEqual(manifest['schema'],'aictrack1.fresh_route.full.training.template.v1')
        for name,digest in manifest['files'].items():
            with self.subTest(member=name):self.assertEqual(cap.sha(full/name),digest)
        self.assertEqual(cap.sha(full/'scripts/capacity_contract.py'),cap.sha(ROOT/'scripts/final_capacity_contract.py'))

if __name__=='__main__':unittest.main(verbosity=2)

"""Essential independent FULL training/recovery and typed admission CPU tests."""
import copy,json,os,subprocess,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'tests')]
import torch,yaml
import full_route_contract as c
import full_route_runtime as runtime
import mainline_guard as guard
import prepare_full_route as prep
from full_recovery_fixture import RecoveryTests,fixture
from test_confirmation_reference import ReceiptEvidenceTests
FULL_H1=Path(r'D:\aicomp\stage2-server-transfer\fv2_full_results_20261008\full\artifacts\experiments\mainline-h1\last.pt')

def choice(route='S'):
    return dict(schema='aictrack1.fresh_route.confirmation.v1',status='complete_positive_fresh_route_confirmation',selected_route=route,selected_arm=c.ROUTES[route],primary_scope='S_P7_vs_C' if route=='S' else 'J5_graph_P7_vs_C',checkpoint_epoch=20,selection_rule='fixed_final20',training_scope='fixed_split_validation',split_sha256=c.FRESH_SPLIT_SHA,h1_sha256=c.FRESH_H1_SHA,source_h1_manifest_sha256=c.FRESH_H1_SOURCE_SHA,test_access=False,used_test_or_online_scores=False,root_risk_reviewed=True,primary_delta_vs_C_pp=.1,natural_delta_vs_C_pp=-.2,training_source_manifest_sha256='a'*64,diagnostic_source_manifest_sha256='b'*64,admission_receipt_sha256='c'*64,evidence={})
def budget():
    now=time.time();return dict(schema='aictrack1.fresh_route.full.budget.v1',root_authorized=True,phase_start_unix=now-1,deadline_unix=now+3600,initial_free_bytes=13013950464,disk_max_bytes=8589934592,min_free_bytes=3221225472)

class ContractTests(unittest.TestCase):
    def test_full_recipe_single_route_and_positive_148695_scope(self):
        for route in c.ROUTES:
            p=prep.full_assets(choice(route),'a'*64,'b'*64);c.validate_assets(p);arm=c.ROUTES[route];raw=c.expected_config(arm,p);c.validate_config(raw,arm,p)
            self.assertEqual(raw['training_scope'],'full_train_fixed_epochs');self.assertEqual(raw['expected_train_count'],148695);self.assertEqual(raw['training']['epochs'],20);self.assertEqual(raw['base_loss']['alpha'],.5 if arm=='S' else .1);self.assertEqual(raw['training']['head_learning_rate'],c.ARMS[arm])
            with self.assertRaises(ValueError):c.expected_config('J5' if arm=='S' else 'S',p)
            bad=copy.deepcopy(p);bad['train_count']=118958
            with self.assertRaises(ValueError):c.validate_assets(bad)
            bad=copy.deepcopy(raw);bad['training']['epochs']=8
            with self.assertRaises(ValueError):c.validate_config(bad,arm,p)

    def test_new_typed_receipt_not_old_naked_J_or_wrong_route(self):
        for route in c.ROUTES:
            v=choice(route);c.validate_confirmation(v)
            for mutation in [{'schema':'aictrack1.p1.fresh_confirmation.v1'},{'status':'selected_recipe_for_confirmation'},{'primary_delta_vs_C_pp':0.},{'root_risk_reviewed':False},{'training_scope':'full_train_fixed_epochs'},{'checkpoint_epoch':8},{'split_sha256':'0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'},{'used_test_or_online_scores':True}]:
                bad=copy.deepcopy(v);bad.update(mutation)
                with self.assertRaises(ValueError):c.validate_confirmation(bad)
            if route=='J5_GRAPH':
                bad=copy.deepcopy(v);bad['primary_scope']='J5_P7_vs_C'
                with self.assertRaises(ValueError):c.validate_confirmation(bad)

    def test_actual_legal_full_H1100_strict_loader(self):
        import mainline_h1_contract as module
        p=prep.full_assets(choice(),'a'*64,'b'*64);p['assets']['h1']['path']=str(FULL_H1)
        with patch.object(module,'assets',return_value=p):
            head,state=module.load_new_fold_head(FULL_H1,feature_dim=512,num_classes=750,expected_clip_weight_sha256=c.CLIP_SHA,expected_class_mapping_sha256=c.MAPPING_SHA,expected_split_sha256=p['assets']['split']['sha256'],forbid_full_train_initializer=False)
            self.assertEqual(c.sha(FULL_H1),c.FULL_H1_SHA);self.assertEqual(state['epoch'],100);self.assertEqual(state['row_binding'],c.read(ROOT/'inputs/legal_full_h1_cpu_acceptance.json')['row_binding']);self.assertEqual(sum(v.numel() for v in head.parameters()),384750)
            with self.assertRaises(ValueError):module.load_new_fold_head(FULL_H1,forbid_full_train_initializer=True)

    def test_explicit_new_full_budget_original_baseline_resume_ledger(self):
        b=budget();c.validate_budget(b)
        for mutation in [{'initial_free_bytes':b['initial_free_bytes']+1},{'root_authorized':False},{'deadline_unix':b['phase_start_unix']},{'min_free_bytes':1},{'phase_start_unix':1791497972.440027,'deadline_unix':1791497972.440028}]:
            with self.assertRaises(ValueError):c.validate_budget(b|mutation)
        with fixture() as folder:
            (folder/'inputs').mkdir();path=folder/'inputs/full_budget.json';path.write_text(json.dumps(b));p={'full_budget_sha256':c.sha(path)}
            with patch('p1_contract.assets',return_value=p):
                self.assertEqual(runtime.bind_budget(folder),b);self.assertEqual(runtime.bind_budget(folder),b)
                path.write_text(json.dumps(b|{'deadline_unix':b['deadline_unix']+1}));p['full_budget_sha256']=c.sha(path)
                with self.assertRaises(ValueError):runtime.bind_budget(folder)
        env=dict(P1_DEADLINE=str(time.time()+3600),P1_INITIAL_FREE=str(13013950464),P1_DISK_MAX=str(8589934592),P1_MIN_FREE=str(3221225472))
        with patch.dict(os.environ,env),patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=5*1024**3)):
            guard._resource_check(0)
            with self.assertRaises(RuntimeError):guard._resource_check(3*1024**3)

    def test_actual_preparation_unique_one_config_and_template_unchanged(self):
        originals={rel:c.sha(ROOT/rel) for rel in prep.verify_template()['files']}
        for route in c.ROUTES:
            with fixture() as folder:
                confirmation=folder/'confirmation.json';confirmation.write_text(json.dumps(choice(route)));boundary=folder/'budget.json';boundary.write_text(json.dumps(budget()));target=folder/'bound'
                # Real evidence checks are separately replayed above; this only fixtures assembly.
                with patch.object(prep,'verify_confirmation',side_effect=c.validate_confirmation):result=prep.prepare(target,confirmation,c.sha(confirmation),boundary,c.sha(boundary))
                self.assertEqual(result['selected_route'],route);self.assertEqual(c.verify_files(target)['status'],'bound');self.assertEqual({p.name for p in (target/'configs/experiment').glob('*.yaml')},{f'p1_full_{c.ROUTES[route]}.yaml'})
                with self.assertRaises(FileExistsError):prep.prepare(target,confirmation,c.sha(confirmation),boundary,c.sha(boundary))
                (target/'scripts/p1_contract.py').write_text('drift')
                with self.assertRaises(ValueError):c.verify_files(target)
        self.assertEqual(originals,{rel:c.sha(ROOT/rel) for rel in originals})

    def test_real_CLI_help_no_gpu_or_server(self):
        for name in ('prepare_full_route.py','run_full.py','train_n1_peft.py'):
            done=subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--help'],capture_output=True,text=True,timeout=30);self.assertEqual(done.returncode,0,name+done.stderr);self.assertIn('usage:',done.stdout)

if __name__=='__main__':
    started=time.monotonic();suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(k) for k in [ContractTests,RecoveryTests,ReceiptEvidenceTests]);result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='aictrack1.fresh_route.full.training.cpu_tests.v1',status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.monotonic()-started,python=sys.executable,torch=torch.__version__,actual_full_h1_sha256=c.sha(FULL_H1),template_manifest_sha256=c.sha(ROOT/'template_manifest.json'),cuda_used=False,server_access=False,actual_model_training_completed=False)
    path=ROOT.parent/'operator_tools/fresh_route_full_training_cpu_tests.json';path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');raise SystemExit(not result.wasSuccessful())

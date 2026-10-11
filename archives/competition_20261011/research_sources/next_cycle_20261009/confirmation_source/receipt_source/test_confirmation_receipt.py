"""CPU-only independent gate tests. No official models or image files loaded."""
import copy,importlib.util,json,sys,unittest,uuid
from pathlib import Path
from write_confirmation_receipt import CLIP,SPLIT,REQUIRED,sha,evidence_files,verify_source,validate_runtime,validate_report,validate_audit,expected_ac_binding
ROOT=Path(__file__).resolve().parent
def evidence():return {name:{'path':'/synthetic/'+name,'sha256':SPLIT if name=='split' else (str(i%10)*64)} for i,name in enumerate(REQUIRED)}
def runtime(suffix,arm,ev):
    return dict(schema='aictrack1.confirmation.runtime.v1',arm=arm,test_access=False,
        source_sha256=ev['training_source']['sha256'],config_sha256=ev['config_'+suffix]['sha256'],split_sha256=ev['split']['sha256'],h1_sha256=ev['h1']['sha256'],
        runtime=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',torch_num_threads=4,torch_num_interop_threads=16))
def report(ev):
    result=dict(schema='aictrack1.confirmation.fixed_last_paired.v1',complete=True,selected_arm='J5',p1_arm='J5',test_access=False,used_test_or_online_scores=False,
        eligibility={'primary_delta_pp':.03,'natural_p7_delta_pp':-.17},protocols=[dict(name=name,p7={'candidate_vs_control':{'delta_vs_baseline_pp':score}}) for name,score in [('balanced_supported_unique',.03),('natural_unique_images',-.17)]])
    for field,name in [('split_sha256','split'),('training_source_manifest_sha256','training_source'),('diagnostic_source_manifest_sha256','diagnostic_source'),('selection_protocol_sha256','selection_protocol'),('frozen_postprocess_sha256','frozen_postprocess')]:result[field]=ev[name]['sha256']
    for kind,suffix in [('control','C'),('candidate','J')]:
        provenance={field:ev[name+'_'+suffix]['sha256'] for field,name in [('identity_sha256','identity'),('checkpoint_sha256','checkpoint'),('checkpoint_audit_sha256','checkpoint_audit')]}
        result[kind]=provenance|{'provenance':dict(provenance)}
    return result
def audit_identity(ev,suffix,arm):
    tensors,numel=(152,87849216) if suffix=='C' else (154,88233966)
    audit=dict(schema='aictrack1.confirmation.fixed_last_diagnostic_audit.v1',complete=True,status='complete_fixed_last20_audit',model_arm=arm,checkpoint_role='last',checkpoint_epoch=20,epoch=20,trainable_tensors=tensors,numel=numel,trainable_numel=numel,
        classifier_head_source='fresh_fold_h1_frozen' if suffix=='C' else 'joint_fixed_last_state',classifier_parameters_restored_equal=True,test_access=False)
    for field,name in [('checkpoint_sha256','checkpoint_'+suffix),('fixed_last_sha256','checkpoint_'+suffix),('config_sha256','config_'+suffix),('split_sha256','split'),('h1_sha256','h1'),('training_source_manifest_sha256','training_source'),('diagnostic_source_manifest_sha256','diagnostic_source')]:audit[field]=ev[name]['sha256']
    identity=dict(holdout_verified=True,training_validation_overlap_rows=0,checkpoint_role='last',test_access=False)
    for field,name in [('checkpoint_sha256','checkpoint_'+suffix),('experiment_config_sha256','config_'+suffix),('split_sha256','split'),('training_source_manifest_sha256','training_source'),('diagnostic_source_manifest_sha256','diagnostic_source')]:identity[field]=ev[name]['sha256']
    return audit,identity
class Tests(unittest.TestCase):
    def test_schema_accepts_actual_frozen_full_validator(self):
        scripts=ROOT.parents[1]/'full_training_source/scripts'
        if not (scripts/'full_confirmation.py').is_file():self.skipTest('independent full validator is not colocated; schema integration was checked on development host')
        sys.path.insert(0,str(scripts))
        spec=importlib.util.spec_from_file_location('independent_full_confirmation',scripts/'full_confirmation.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ev=evidence();gate=validate_report(report(ev),ev,'J5');r=validate_runtime(runtime('C','C',ev),runtime('J','J5',ev),ev,'J5')
        value=dict(schema='aictrack1.p1.fresh_confirmation.v1',status='positive_confirmed_for_full_training',selected_arm='J5',split_seed=20261009,split_sha256=SPLIT,
            h1_sha256=ev['h1']['sha256'],h1_receipt_sha256=ev['h1_receipt']['sha256'],confirmation_training_source_manifest_sha256=ev['training_source']['sha256'],diagnostic_source_manifest_sha256=ev['diagnostic_source']['sha256'],selection_protocol_sha256=ev['selection_protocol']['sha256'],frozen_postprocess_sha256=ev['frozen_postprocess']['sha256'],train_count=118958,validation_count=29737,
            train_query_overlap=0,old_query_overlap=0,test_access=False,used_test_or_online_scores=False,initialization=dict(clip_sha256=CLIP,h1_epoch=100,h1_scope='fold_train_fixed_final_epochs'),runtime=r,decision_gate=gate,root_review=dict(status='pass',reviewer='root',reason='synthetic schema only; no actual model PASS'),evidence=ev,
            arms={arm:dict(epoch=20,checkpoint_role='last')|{field:ev[name+'_'+suffix]['sha256'] for field,name in [('checkpoint_sha256','checkpoint'),('config_sha256','config'),('identity_sha256','identity'),('checkpoint_audit_sha256','checkpoint_audit')]} for arm,suffix in [('C','C'),('J5','J')]})
        self.assertIs(module.validate_confirmation(value),value)
    def test_ac_indices_are_manifest_global_selected_train_not_renumbered_or_full(self):
        rows=[dict(sample_id=s,internal_class_index=str(i)) for i,s in enumerate(['a','b','c'])]
        indices,labels=expected_ac_binding(rows,{'a':'train','b':'validation','c':'train'})
        self.assertEqual(indices,[0,2]);self.assertEqual(labels,[0,2]);self.assertNotEqual(indices,list(range(2)))
        with self.assertRaises(ValueError):expected_ac_binding(rows,{'a':'train','c':'train'})
    def test_primary_positive_accepts_negative_natural_and_no_ci_lower_gate(self):
        gate=validate_report(report(evidence()),evidence(),'J5');self.assertEqual(gate['status'],'pass');self.assertEqual(gate['natural_p7_delta_pp'],-.17);self.assertFalse(gate['stable_gain_claim'])
    def test_nonpositive_primary_and_nonfinite_natural_rejected(self):
        for score in (0.,-.01):
            r=report(evidence());r['eligibility']['primary_delta_pp']=score;r['protocols'][0]['p7']['candidate_vs_control']['delta_vs_baseline_pp']=score
            with self.assertRaises(ValueError):validate_report(r,evidence(),'J5')
        r=report(evidence());r['eligibility']['natural_p7_delta_pp']=float('nan');r['protocols'][1]['p7']['candidate_vs_control']['delta_vs_baseline_pp']=float('nan')
        with self.assertRaises(ValueError):validate_report(r,evidence(),'J5')
    def test_old_split_protocol_or_old_report_rejected(self):
        for key,value in [('schema','aictrack1.p1.fixed_last_paired.v1'),('split_sha256','0'*64),('selection_protocol_sha256','0'*64),('used_test_or_online_scores',True)]:
            r=report(evidence());r[key]=value
            with self.assertRaises(ValueError):validate_report(r,evidence(),'J5')
    def test_pair_summary_and_alias_conflict_rejected(self):
        r=report(evidence());r['eligibility']['primary_delta_pp']=9.
        with self.assertRaises(ValueError):validate_report(r,evidence(),'J5')
        r=report(evidence());r['candidate']['identity_sha256']='f'*64
        with self.assertRaises(ValueError):validate_report(r,evidence(),'J5')
    def test_actual_runtime_binding_and_threads(self):
        ev=evidence();C=runtime('C','C',ev);J=runtime('J','J5',ev);self.assertTrue(validate_runtime(C,J,ev,'J5')['equal_between_arms'])
        for change in ['threads','source','interop']:
            bad=copy.deepcopy(J)
            if change=='threads':bad['runtime']['torch_num_threads']=64
            if change=='source':bad['source_sha256']='f'*64
            if change=='interop':bad['runtime']['torch_num_interop_threads']=17
            with self.assertRaises(ValueError):validate_runtime(C,bad,ev,'J5')
    def test_exact154_new_head_and_fixedlast_audit(self):
        ev=evidence();a,i=audit_identity(ev,'J','J5');validate_audit(a,i,ev,'J','J5')
        for field,value in [('trainable_tensors',152),('classifier_head_source','fresh_fold_h1_frozen'),('classifier_parameters_restored_equal',False),('epoch',8),('checkpoint_role','best')]:
            bad=copy.deepcopy(a);bad[field]=value
            with self.assertRaises(ValueError):validate_audit(bad,i,ev,'J','J5')
    def test_c152_frozen_h1_and_wrong_sidecar_rejected(self):
        ev=evidence();a,i=audit_identity(ev,'C','C');validate_audit(a,i,ev,'C','C')
        for field,value in [('split_sha256','f'*64),('training_validation_overlap_rows',1),('holdout_verified',False)]:
            bad=copy.deepcopy(i);bad[field]=value
            with self.assertRaises(ValueError):validate_audit(a,bad,ev,'C','C')
    def test_actual_files_hashed_and_drift_rejected(self):
        directory=ROOT/'test_fixtures'/uuid.uuid4().hex;directory.mkdir(parents=True)
        request={'evidence':{}}
        for name in REQUIRED:
            path=directory/(name+'.json');path.write_text(json.dumps({'synthetic':True,'name':name})+'\n');request['evidence'][name]={'path':str(path),'sha256':sha(path)}
        ev=evidence_files(request);self.assertEqual(len(ev),19)
        Path(ev['split']['path']).write_text('changed\n')
        with self.assertRaises(ValueError):evidence_files(request)
        file=directory/'module.py';file.write_text('VALUE=1\n');manifest=directory/'source.json';manifest.write_text(json.dumps({'files':{'module.py':sha(file)}}))
        verify_source({'path':str(manifest),'sha256':sha(manifest)});file.write_text('VALUE=2\n')
        with self.assertRaises(ValueError):verify_source({'path':str(manifest),'sha256':sha(manifest)})
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    path=ROOT/('CPU_TESTS_'+sha(__file__)[:16]+'.json');path.write_text(json.dumps(dict(schema='aictrack1.confirmation.receipt_cpu_tests.v1',tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),model_training=False,gpu_access=False,test_access=False),indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)

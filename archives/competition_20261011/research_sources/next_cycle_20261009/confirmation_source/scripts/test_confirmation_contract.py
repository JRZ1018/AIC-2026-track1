"""Bounded CPU tests: connected groups, old-query exclusion, fixed selection and state binding."""
from __future__ import annotations
import ast,copy,dataclasses,json,sys,unittest,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aictrack1.data.manifest import SampleRecord
from prepare_confirmation_assets import connected_groups,build_split,audit_split,sha
from confirmation_contract import HEAD_CONFIG,SCOPE,SELECTION,row_binding,validate_h1
from prepare_confirmation_visual import validate_selection,create_visual_source
def record(i,label=0,exact='',near='',image_hash=''):
    return SampleRecord(sample_id=f's{i:03d}',relative_path=f'{label:04d}/image{i}.jpg',source_split='train',observed_label=f'{label:04d}',internal_class_index=label,official_class_id=f'{label:04d}',decode_status='ok',duplicate_group=exact,near_duplicate_group=near,sha256=image_hash)
def selection():
    return dict(schema='aictrack1.p1.full_selection.v1',status='selected_recipe_for_confirmation',training_complete=True,used_test_or_online_scores=False,decision_basis='current_stage_train_derived_validation',training_scope='fixed_split_validation',checkpoint_role='last',checkpoint_epoch=20,train_count=118958,validation_count=29737,head_learning_rate=5e-6,test_access=False,selection_split_sha256='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c',selection_rule='fixed_final20',selected_arm='J5',selected_fold_checkpoint_sha256='a'*64,selected_fold_config_sha256='b'*64,diagnostic_source_manifest_sha256='c'*64,selection_protocol_sha256='d'*64,candidates=[dict(arm='J5',epoch=20,checkpoint_role='last',checkpoint_sha256='a'*64)])
def h1_state():
    import torch
    rows=[dict(sample_id='a',target='0',split='train'),dict(sample_id='b',target='1',split='validation')]
    binding=row_binding(rows);assets=dict(counts=dict(train=1,validation=1),assets=dict(split=dict(sha256='e'*64),manifest=dict(sha256='f'*64)))
    state=dict(training_scope=SCOPE,epoch=100,config=HEAD_CONFIG,selection_rule=SELECTION,head={'kind':'linear'},row_binding=binding,
        feature_cache_metadata=dict(training_scope=SCOPE,training_record_count=1,split_sha256='e'*64),best_model=None,
        history=[dict(epoch=i,train_loss=.1,learning_rate=.01) for i in range(1,101)],model={'linear.weight':torch.zeros(750,512),'linear.bias':torch.zeros(750)},competition_stage='stage2',train_manifest_sha256='f'*64,split_sha256='e'*64,class_mapping_sha256='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879',clip_weight_sha256='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af')
    state.update(optimizer={'param_groups':[{'params':[0,1]}],'state':{0:{},1:{}}},scheduler={'last_epoch':100},scaler={},python_rng_state=None,numpy_rng_state=None,torch_rng_state=None,cuda_rng_state_all=[],loss_config={},class_weights=None,best_top1=float('-inf'))
    return state,assets,binding
class Tests(unittest.TestCase):
    def test_connected_bridge_preserves_all_relations(self):
        records=[record(0,exact='E',near='N1'),record(1,exact='E',near='N2'),record(2,near='N2'),record(3,image_hash='same'),record(4,image_hash='same')]
        groups,info=connected_groups(records)
        self.assertEqual(len({groups[f's{i:03d}'] for i in (0,1,2)}),1);self.assertEqual(groups['s003'],groups['s004']);self.assertEqual(info['component_count'],2)
    def test_new_query_excludes_connected_old_query(self):
        records=[record(i) for i in range(20)];old={r.sample_id:('validation' if i<4 else 'train') for i,r in enumerate(records)}
        assigned,groups,_,forbidden=build_split(records,old)
        self.assertEqual(sum(v=='validation' for v in assigned.values()),4)
        self.assertTrue(all(assigned[r.sample_id]=='train' for r in records[:4]));audit_split(records,assigned,groups,forbidden)
    def test_insufficient_remaining_pool_no_forbidden_fallback(self):
        records=[record(i,near='all') for i in range(5)];old={r.sample_id:('validation' if i==0 else 'train') for i,r in enumerate(records)}
        assigned,_,_,_=build_split(records,old)
        self.assertEqual(set(assigned.values()),{'train'})
    def test_leaking_connected_assignment_rejected(self):
        records=[record(0,exact='E'),record(1,exact='E')];groups,_=connected_groups(records)
        with self.assertRaisesRegex(ValueError,'connected group'):audit_split(records,{'s000':'train','s001':'validation'},groups,set())
    def test_rows_digest_binds_label_order_and_partition(self):
        a=[dict(sample_id='a',target='0',split='train'),dict(sample_id='b',target='1',split='validation')]
        b=copy.deepcopy(a);b[0]['target']='1';self.assertNotEqual(row_binding(a)['train_rows_sha256'],row_binding(b)['train_rows_sha256'])
        self.assertNotEqual(row_binding(a)['all_rows_sha256'],row_binding(a[::-1])['all_rows_sha256'])
    def test_actual_h1_fixed100_contract_and_full_scope_rejection(self):
        state,assets,binding=h1_state();validate_h1(state,assets,binding)
        state['training_scope']='full_train_fixed_epochs'
        with self.assertRaises(ValueError):validate_h1(state,assets,binding)
    def test_h1_best_validation_history_and_wrong_rows_rejected(self):
        state,assets,binding=h1_state()
        for change in ('best','history','rows'):
            bad=copy.deepcopy(state)
            if change=='best':bad['best_model']={}
            if change=='history':bad['history'][0]['val_top1']=.7
            if change=='rows':bad['row_binding']['train_rows_sha256']='0'*64
            with self.assertRaises(ValueError):validate_h1(bad,assets,binding)
    def test_independent_winner_last20_only(self):
        self.assertEqual(validate_selection(selection()),'J5')
        for change in ('best','epoch8','newsplit','test','selected_sha'):
            bad=selection()
            if change=='best':bad['candidates'][0]['checkpoint_role']='best'
            if change=='epoch8':bad['candidates'][0]['epoch']=8
            if change=='newsplit':bad['selection_split_sha256']='0'*64
            if change=='test':bad['test_access']=True
            if change=='selected_sha':bad['selected_fold_checkpoint_sha256']='e'*64
            with self.assertRaises(ValueError):validate_selection(bad)
    def test_canonical_queue_original_bytes_preserved(self):
        original=json.loads((ROOT/'clone_receipt.json').read_text())['source_files']
        rel='src/aictrack1/engine/training_queue.py';self.assertEqual(sha(ROOT/rel),original[rel])
    def test_visual_source_real_builder_only_two_matched_configs(self):
        fixtures=ROOT/'test_fixtures';fixtures.mkdir(exist_ok=True)
        source=fixtures/('visual_'+uuid.uuid4().hex);receipt=fixtures/'selection_input.json'
        receipt.write_text(json.dumps(selection())+'\n')
        audit=json.loads((ROOT/'inputs/confirmation_assets.json').read_text())
        official={'clip':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af','manifest':'1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'}
        assets={n:dict(path='/synthetic/'+n,sha256=official.get(n,'1'*64)) for n in ['manifest','split','class_mapping','clip','h1','label_conflicts']}
        assets['split']['sha256']=audit['assets']['split']['sha256']
        payload=dict(schema='aictrack1.p1.assets.v1',status='verified_fold_assets',competition_stage='stage2',train_count=118958,validation_count=29737,num_classes=750,
            h1_training_scope='fold_train_fixed_final_epochs',h1_training_count=118958,h1_validation_used_for_selection=False,h1_epoch=100,terminal_cache_parity_top1=.5,
            assets=assets,dataset_root='/synthetic/official_stage2',canonical_lock='/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json',training_seed=20261008,split_seed=20261009,
            h1_row_binding=audit['h1_row_binding'],selected_arm='J5',confirmation=True,test_access=False,control={'role':'fresh_official_clip_new_fold_h1_fixed20_no_checkpoint_resume'})
        create_visual_source(source,payload,'J5',receipt,sha(receipt),'2'*64)
        import yaml
        C=yaml.safe_load((source/'configs/experiment/p1_C.yaml').read_text())['experiment'];J=yaml.safe_load((source/'configs/experiment/p1_J5.yaml').read_text())['experiment']
        self.assertFalse((source/'configs/experiment/p1_J20.yaml').exists());self.assertEqual(C['training']['head_learning_rate'],0.);self.assertEqual(J['training']['head_learning_rate'],5e-6)
        for x in (C,J):
            for k in ['run_id','output_dir']:x.pop(k)
            x['training'].pop('head_learning_rate')
        self.assertEqual(C,J)
        manifest=json.loads((source/'source_manifest.json').read_text())
        self.assertTrue(all(sha(source/rel)==value for rel,value in manifest['files'].items()))
        for path in (source/'scripts').glob('*.py'):ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        self.assertIn('OMP_NUM_THREADS', (source/'scripts/run_p1.py').read_text());self.assertIn('torch.get_num_threads()!=4',(source/'scripts/mainline_guard.py').read_text())
        original=json.loads((ROOT/'clone_receipt.json').read_text())['source_files']
        self.assertEqual(sha(source/'scripts/train_n1_peft.py'),original['scripts/train_n1_peft.py']);self.assertEqual(sha(source/'src/aictrack1/engine/training_queue.py'),original['src/aictrack1/engine/training_queue.py'])
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    from confirmation_contract import write_new
    write_new(ROOT/('cpu_contract_tests_'+sha(__file__)[:16]+'.json'),dict(schema='aictrack1.confirmation.cpu.tests.v1',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),gpu_access=False,model_training=False,test_access=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)

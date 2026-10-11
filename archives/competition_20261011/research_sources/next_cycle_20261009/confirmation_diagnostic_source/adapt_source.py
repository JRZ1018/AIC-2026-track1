"""One-time adaptation recorded in source inventory; refuses a frozen target."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
if (ROOT/'source_manifest.json').exists():raise FileExistsError('already frozen')
def replace(text,old,new):
 if old not in text:raise ValueError('missing adaptation anchor '+old[:80])
 return text.replace(old,new)
p=ROOT/'run_p1_diagnostic.py';s=p.read_text()
s=replace(s,'from p1_diag_contract import (ARMS, HEAD_KEYS, H1_SHA, JOINT_KEYS, JOINT_NUMEL,\n    MAPPING_SHA, ROOT, SPLIT_SHA, TRAINING_MANIFEST_SHA, VIEW_PROTOCOL,\n    save_json, sha, validate_checkpoint_metadata, validate_joint_state,\n    validate_optimizer_state, verify_owned_sources)',
'''# Matched confirmation runtime is established before importing torch-dependent contracts.
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from confirmation_diag_contract import (ARMS, HEAD_KEYS, JOINT_KEYS, JOINT_NUMEL,
    VISUAL_KEYS, VISUAL_NUMEL, MAPPING_SHA, ROOT, SPLIT_SHA, VIEW_PROTOCOL,
    save_json, sha, validate_checkpoint_metadata, validate_state,
    validate_optimizer_state, verify_owned_sources, verify_training_binding, confirmation_rows)
TRAINING_MANIFEST_SHA = None
REGISTERED = None
SOURCE_MANIFEST = None''')
s=replace(s,"str(ROOT / 'run_p1_diagnostic.py')","str(ROOT / 'run_confirmation_diagnostic.py')")
s=replace(s,'def dependency_probe(training_root):','def dependency_probe(training_root, expected_source_sha):')
s=replace(s,"    if sha(training_root / 'source_manifest.json') != TRAINING_MANIFEST_SHA:\n        raise ValueError('dependency probe requires frozen 2fee training source')", "    declarations, source = verify_training_binding(training_root, expected_source_sha)")
s=replace(s,'    for arm in ARMS:\n        config',"    for arm in ('C', source['selected_arm']):\n        config")
s=replace(s,"training_root / 'configs/experiment/p1_J5.yaml',\n        training_root / 'artifacts/experiments/P1_J5/last.pt'", "training_root / 'configs/experiment' / ('p1_'+source['selected_arm']+'.yaml'),\n        training_root / 'artifacts/experiments' / ('P1_'+source['selected_arm']) / 'last.pt'")
s=replace(s,"frozen_training_source_sha256=TRAINING_MANIFEST_SHA", "frozen_training_source_sha256=expected_source_sha")
s=replace(s,"schema='aictrack1.p1.cpu_entry_dependency_check.v1'", "schema='aictrack1.confirmation.cpu_entry_dependency_check.v1'")
s=replace(s,"    return partitions\n", """    complete, connected = confirmation_rows()
    for part in ('train', 'validation'):
        records = partitions[part]
        expected = connected[part]
        if ([r['sample_id'] for _, r in records] != [r['sample_id'] for r in expected]
                or [int(r['internal_class_index']) for _, r in records] != [int(r['target']) for r in expected]):
            raise ValueError('fresh connected row order/labels differ from official manifest')
    return partitions
""")
s=replace(s,'config_sha=sha(config), last_sha=sha(checkpoint))','config_sha=sha(config), last_sha=sha(checkpoint), registered=registered)')
s=replace(s,"    if sha(experiment['head_checkpoint']) != H1_SHA:\n        raise ValueError('H1 initializer identity mismatch')", "    if sha(experiment['head_checkpoint']) != registered['assets']['h1']['sha256']:\n        raise ValueError('fresh H1 initializer identity mismatch')")
s=replace(s,'audit.update(validate_joint_state(state[\'trainable_state\'], visual_names=visual_names))',"audit.update(validate_state(state['trainable_state'], arm=CURRENT_ARM, visual_names=visual_names))")
s=replace(s,"    if set(names) != set(params) or len(names) != JOINT_KEYS or audit['trainable_numel'] != JOINT_NUMEL:\n        raise ValueError('joint state must restore all 154 model parameters exactly')", """    expected_names = set(visual_names) if CURRENT_ARM == 'C' else set(params)
    expected_numel = VISUAL_NUMEL if CURRENT_ARM == 'C' else JOINT_NUMEL
    if set(names) != expected_names or audit['trainable_numel'] != expected_numel:
        raise ValueError('fresh C visual152 / J joint154 exact state required')
    if CURRENT_ARM == 'C':
        fresh_head, _ = load_new_fold_head(Path(experiment['head_checkpoint']), feature_dim=512, num_classes=750,
            expected_clip_weight_sha256=registered['assets']['clip']['sha256'],
            expected_class_mapping_sha256=MAPPING_SHA, expected_split_sha256=SPLIT_SHA, expected_provenance=None)
        for key, value in fresh_head.named_parameters():
            if not torch.equal(params['classifier.'+key].detach(), value.detach()):
                raise ValueError('C classifier differs from actual fresh train-only H1')
""")
s=replace(s,'validate_optimizer_state(state, names, visual_names=visual_names)', 'validate_optimizer_state(state, names, visual_names=visual_names, arm=CURRENT_ARM)')
s=replace(s,"    confidence = state['confidence_weighting']", """    confidence = state['confidence_weighting']
    from aictrack1.methods.confidence import ClasswiseHistoryWeighting
    confidence_raw = experiment['confidence_weighting']
    restored_ac = ClasswiseHistoryWeighting(
        torch.tensor([i for i, _ in fold['train']]),
        torch.tensor([int(r['internal_class_index']) for _, r in fold['train']]),
        **{k:confidence_raw[k] for k in ('momentum','history_mix','min_weight','magnitude_mix',
            'tail_class_fraction','tail_min_mean_weight','min_effective_mass_per_class') if k in confidence_raw})
    restored_ac.load_state_dict(confidence)
    if any(isinstance(v, torch.Tensor) and v.is_floating_point() and not bool(torch.isfinite(v).all())
           for v in confidence.values()):
        raise ValueError('nonfinite full AC state')
""")
s=replace(s,"schema='aictrack1.p1.fixed_last_diagnostic_audit.v1', p1_arm=CURRENT_ARM,\n                 classifier_head_source='joint_fixed_last_state',", "schema='aictrack1.confirmation.fixed_last_diagnostic_audit.v1', p1_arm=CURRENT_ARM, model_arm=CURRENT_ARM,\n                 complete=True, status='complete_fixed_last20_audit',\n                 classifier_head_source=('fresh_fold_h1_frozen' if CURRENT_ARM=='C' else 'joint_fixed_last_state'),")
s=replace(s,'inference_optimizer_steps=0, historical_control_cache_reused=True)',"inference_optimizer_steps=0, historical_control_cache_reused=False, test_access=False,\n                 selection_receipt_sha256=SOURCE_MANIFEST['selection_receipt_sha256'],\n                 source_h1_manifest_sha256=SOURCE_MANIFEST['source_h1_manifest_sha256'])")
s=replace(s,"    with row_path.open('x'", "    _, connected = confirmation_rows()\n    connected_by_id = {r['sample_id']:r for r in connected['validation']}\n    with row_path.open('x'")
s=replace(s,"            group = ('near:' + near) if near else ('exact:' + exact) if exact else ('sample:' + row['sample_id'])", "            group = connected_by_id[row['sample_id']]['group']")
s=replace(s,'h1_sha256=H1_SHA',"h1_sha256=registered['assets']['h1']['sha256']")
s=replace(s,"report_sha256=sha(output / 'report.json'), p1_arm=CURRENT_ARM,\n        joint_trainable_tensors=JOINT_KEYS, joint_trainable_numel=JOINT_NUMEL,\n        classifier_head_source='joint_fixed_last_state',", "report_sha256=sha(output / 'report.json'), p1_arm=CURRENT_ARM, model_arm=CURRENT_ARM,\n        trainable_tensors=audit['trainable_tensors'], trainable_numel=audit['trainable_numel'],\n        joint_trainable_tensors=audit['trainable_tensors'], joint_trainable_numel=audit['trainable_numel'],\n        classifier_head_source=audit['classifier_head_source'], selected_arm=SOURCE_MANIFEST['selected_arm'],\n        selection_receipt_sha256=SOURCE_MANIFEST['selection_receipt_sha256'],\n        source_h1_manifest_sha256=SOURCE_MANIFEST['source_h1_manifest_sha256'],")
s=replace(s,"group_source='official manifest near_duplicate_group, then duplicate_group; sample ID only for true singleton'", "group_source='fresh confirmation rows; connected exact SHA/exact-group/near-group union; no old-query connected group'")
s=replace(s,'global TRAIN_ROOT, CURRENT_ARM, OUTPUT, SOURCE_SHA, SAVE_FEATURES','global TRAIN_ROOT, CURRENT_ARM, OUTPUT, SOURCE_SHA, SAVE_FEATURES, TRAINING_MANIFEST_SHA, REGISTERED, SOURCE_MANIFEST')
s=replace(s,"    parser.add_argument('--arm'", "    parser.add_argument('--expected-training-source-sha256', required=True)\n    parser.add_argument('--arm'")
s=replace(s,'dependency_probe(args.training_root)','dependency_probe(args.training_root, args.expected_training_source_sha256)')
s=replace(s,"    TRAIN_ROOT = args.training_root.resolve()", """    TRAIN_ROOT = args.training_root.resolve()
    import torch
    torch.set_num_threads(4)
    TRAINING_MANIFEST_SHA = args.expected_training_source_sha256
    REGISTERED, SOURCE_MANIFEST = verify_training_binding(TRAIN_ROOT, TRAINING_MANIFEST_SHA, runtime_assets=True)
    if args.arm not in ('C', SOURCE_MANIFEST['selected_arm']):
        raise ValueError('only new C and the preselected J may be diagnosed')
    if str(TRAIN_ROOT) != '/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009/prepared_visual':
        raise ValueError('production inference must bind canonical fresh prepared_visual root')
""")
s=replace(s,"    if registered != json.loads((ROOT / 'inputs' / 'p1_assets.json').read_text(encoding='utf-8')):\n        raise ValueError('diagnostic and training asset declarations differ')", "    if registered != REGISTERED:\n        raise ValueError('fresh declarations changed during preflight')")
(ROOT/'run_confirmation_diagnostic.py').write_text(s,encoding='utf-8');p.unlink()

# Compare path adopts fresh positive binding; query freezing has no checkpoint dependency.
p=ROOT/'compare_p1.py';s=p.read_text()
s=replace(s,'from p1_diag_contract import (ROOT, CONTROL_CACHE_SHA, CONTROL_LAST_SHA,\n    load_cache, save_json, sha, validate_pair, verify_owned_sources)', 'from confirmation_diag_contract import (ROOT, SPLIT_SHA, ROWS_SHA, confirmation_rows,\n    load_cache, save_json, sha, validate_pair, verify_owned_sources)')
s=replace(s,"    identity = dict(schema='aictrack1.p1.frozen_queries.v1', control_bundle_sha256=CONTROL_CACHE_SHA,\n        control_last_sha256=CONTROL_LAST_SHA, control_ids_sha256=control['identity']['sample_ids_order_sha256'],", "    identity = dict(schema='aictrack1.confirmation.frozen_queries.v1', split_sha256=SPLIT_SHA,\n        confirmation_rows_sha256=ROWS_SHA, control_ids_sha256=control['identity']['sample_ids_order_sha256'],")
s=replace(s,"identity.get('schema') != 'aictrack1.p1.frozen_queries.v1'\n            or identity['control_bundle_sha256'] != CONTROL_CACHE_SHA\n            or identity['control_last_sha256'] != CONTROL_LAST_SHA", "identity.get('schema') != 'aictrack1.confirmation.frozen_queries.v1'\n            or identity['split_sha256'] != SPLIT_SHA\n            or identity['confirmation_rows_sha256'] != ROWS_SHA")
start=s.index('def eligibility(results):');end=s.index('\n\ndef main():',start)
s=s[:start]+'''def eligibility(results):
    delta=next(r for r in results if r['name']=='balanced_supported_unique')['p7']['candidate_vs_control']['delta_vs_baseline_pp']
    natural=next(r for r in results if r['name']=='natural_unique_images')['p7']['candidate_vs_control']['delta_vs_baseline_pp']
    ci=next(r for r in results if r['name']=='balanced_supported_unique')['p7']['candidate_vs_control'].get('group_bootstrap_95ci_delta_pp')
    return dict(status=('fresh_registered_direction_positive' if delta>0 else 'fresh_registered_direction_not_positive'),
        primary_delta_pp=delta,natural_p7_delta_pp=natural,primary_metric='balanced_supported_unique P7 observed-label agreement',
        registered_rule='preselected unique J: fresh same-protocol primary delta > 0; complete provenance and matched runtime; root risk review',
        natural_nonnegative_hard_gate=False,ci_lower_bound_positive_hard_gate=False,
        stable_gain_claim=False,formal_test_frozen=False,low_support_classes_do_not_alone_veto=True)
''' +s[end:]
s=replace(s,"    control = load_cache(args.control_bundle, args.control_identity)\n    if control['identity']['bundle_sha256'] != CONTROL_CACHE_SHA or control['identity']['checkpoint_sha256'] != CONTROL_LAST_SHA:\n        raise ValueError('wrong control cache: best and last8 are forbidden')", "    control = load_cache(args.control_bundle, args.control_identity)\n    if control['identity']['model_arm'] != 'C':\n        raise ValueError('fresh C control required')")
s=replace(s,"schema='aictrack1.p1.fixed_last_paired.v1'", "schema='aictrack1.confirmation.fixed_last_paired.v1'")
s=replace(s,"p1_arm=candidate['identity']['p1_arm'], candidate=candidate['provenance'], control=control['provenance'],", "p1_arm=candidate['identity']['p1_arm'], selected_arm=candidate['identity']['model_arm'],\n        split_sha256=SPLIT_SHA, h1_sha256=candidate['identity']['h1_sha256'],\n        training_source_manifest_sha256=candidate['identity']['training_source_manifest_sha256'],\n        frozen_postprocess_sha256=sha(ROOT/'frozen_postprocess.json'), used_test_or_online_scores=False,\n        candidate=dict(candidate['provenance'],provenance=candidate['provenance']),\n        control=dict(control['provenance'],provenance=control['provenance']),")
s=replace(s,'not a full750 metric or independent new fold','not a full750 metric; independent new confirmation fold')
(ROOT/'compare_confirmation.py').write_text(s,encoding='utf-8');p.unlink()

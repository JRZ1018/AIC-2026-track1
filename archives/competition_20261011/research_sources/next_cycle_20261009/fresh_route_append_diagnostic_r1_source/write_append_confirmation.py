"""Validate real appended-stage provenance, then use the unchanged canonical typed writer."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from append_diagnostic_contract import (validate_append_budget, actual_evidence, require, sha, read,
    ORIGINAL_START, ORIGINAL_DEADLINE, ORIGINAL_BASELINE, DISK_MAX, MIN_FREE)
from fresh_diag_contract import verify_owned_sources, save_json
from fresh_diag_contract import SPLIT_SHA, H1_SHA
from write_fresh_route_confirmation import build_confirmation

def validate_append_evidence(report, items, context):
    """Extra provenance is semantically checked here; the canonical writer retains all real hashes."""
    for name, wanted in [('original_fresh_budget', context['original_fresh_budget']),
                         ('appended_stage_budget', context['appended_stage_budget'])]:
        require(items.get(name) == wanted, 'exact appended/original budget evidence required: ' + name)
        actual_evidence(items[name], name)
    require(report.get('append_stage_context') == context, 'paired graph report must disclose this exact appended stage')
    completion = report.get('appended_stage_completed_at_unix')
    require(isinstance(completion, (int, float)) and math.isfinite(completion)
            and context['phase_start_unix'] <= completion < context['deadline_unix'],
            'paired computation must actually complete inside the one appended window')
    identities = {}
    for suffix, arm in [('C', 'C'), ('candidate', 'J5')]:
        identity = actual_evidence(items.get('identity_' + suffix), 'identity_' + suffix)
        audit = actual_evidence(items.get('checkpoint_audit_' + suffix), 'checkpoint_audit_' + suffix)
        runtime = actual_evidence(items.get('runtime_' + suffix), 'runtime_' + suffix)
        for value in (identity, audit, runtime):
            require(value.get('append_stage_context') == context, 'same appended source/window metadata on actual restore/runtime/identity')
            completed = value.get('appended_stage_completed_at_unix')
            require(isinstance(completed, (int, float)) and math.isfinite(completed)
                    and context['phase_start_unix'] <= completed < context['deadline_unix'],
                    'actual 4V cache must complete inside appended stage')
        require(identity.get('checkpoint_role') == 'last' and identity.get('checkpoint_epoch') == 20
                and identity.get('model_arm') == arm and identity.get('selected_route') == 'J5_GRAPH',
                'only the actual complete C/J5 fixed-last20 route')
        require(identity.get('training_source_manifest_sha256') == context['training_source']['sha256']
                and identity.get('experiment_config_sha256') == context['configs'][arm]['sha256'],
                'appended restored model origin/source/config mismatch')
        identities[arm] = identity
    review = actual_evidence(items.get('root_review'), 'root_review')
    require(review.get('appended_stage_budget_sha256') == context['appended_stage_budget']['sha256']
            and review.get('original_fresh_budget_sha256') == context['original_fresh_budget']['sha256']
            and review.get('acknowledges_completion_after_original_deadline') is True,
            'root risk review must explicitly acknowledge this appended window and unchanged original budget')
    origins = context['checkpoint_origins']
    require(identities['C']['checkpoint_sha256'] == origins['C']['sha256'], 'C20 baseline must remain the same actual checkpoint')
    if origins['J5']['epoch'] == 20:
        require(identities['J5']['checkpoint_sha256'] == origins['J5']['sha256'],
                'already-complete J5 checkpoint must not be retrained or replaced')
    verify_continuation_chain(items, context, identities['J5']['checkpoint_sha256'])
    require(review.get('continuation_operator_source_manifest_sha256') == items['continuation_operator_source']['sha256'],
            'root must review the actual concrete continuation operator source SHA')
    chain = {name: items[name] for name in CHAIN_NAMES}
    for suffix in ('C', 'candidate'):
        for prefix in ('identity_', 'checkpoint_audit_', 'runtime_'):
            value = actual_evidence(items[prefix + suffix], prefix + suffix)
            require(value.get('continuation_execution_chain') == chain, 'raw caches must bind the reviewed execution chain')
    return True

CHAIN_NAMES = ('continuation_operator_source', 'partial_restore_audit', 'terminal_restore_audit', 'continuation_completion')

def verify_continuation_chain(items, context, final_sha):
    for name in CHAIN_NAMES:
        require(name in items, 'partial J5 requires the actual full continuation chain: ' + name)
    operator = actual_evidence(items['continuation_operator_source'], 'continuation_operator_source')
    audit = actual_evidence(items['partial_restore_audit'], 'partial_restore_audit')
    terminal = actual_evidence(items['terminal_restore_audit'], 'terminal_restore_audit')
    done = actual_evidence(items['continuation_completion'], 'continuation_completion')
    require(operator.get('schema') == 'aictrack1.fresh_route.continuation.training.source.v1'
            and isinstance(operator.get('files'), dict) and bool(operator['files']), 'actual frozen continuation operator inventory')
    operator_root = Path(items['continuation_operator_source']['path']).resolve().parent
    for name, digest in operator['files'].items():
        member = (operator_root / name).resolve()
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and member.is_relative_to(operator_root)
                and sha(member) == digest, 'actual frozen continuation operator member drift')
    partial = context['checkpoint_origins']['J5']['epoch'] < 20
    resume_sha = context['checkpoint_origins']['J5']['sha256'] if partial else None
    for value, name in [(audit, 'partial restore audit'), (terminal, 'terminal restore audit'), (done, 'continuation completion')]:
        require(value.get('shared_budget_sha256') == context['appended_stage_budget']['sha256']
                and value.get('operator_source_manifest_sha256') == items['continuation_operator_source']['sha256']
                and value.get('original_budget_sha256') == context['original_fresh_budget']['sha256']
                and value.get('resume_checkpoint_sha256') == resume_sha,
                'actual source/budget/partial-checkpoint lineage mismatch: ' + name)
    require(audit.get('schema') == 'aictrack1.fresh_route.continuation.cpu_checkpoint_audit.v1'
            and audit.get('complete') is True and audit.get('source_manifest_sha256') == context['training_source']['sha256']
            and audit.get('original_diagnostic_source_manifest_sha256') == '554d7c8f839d4ab6fee1ce427c5e2b9b73435d4766ac5347794ceecace53fbc2'
            and audit.get('selected_route') == 'J5_GRAPH' and audit.get('test_access') is False
            and audit.get('cuda_used') is False and audit.get('model_forwards') == 0 and audit.get('optimizer_steps') == 0,
            'actual immutable two-model CPU pre-resume audit required')
    for arm in ('C', 'J5'):
        state = audit.get(arm, {})
        epoch = context['checkpoint_origins'][arm]['epoch']
        require(state.get('checkpoint_sha256') == context['checkpoint_origins'][arm]['sha256']
                and state.get('checkpoint_epoch') == epoch and state.get('config_sha256') == context['configs'][arm]['sha256']
                and state.get('source_manifest_sha256') == context['training_source']['sha256']
                and state.get('split_sha256') == SPLIT_SHA and state.get('h1_sha256') == H1_SHA
                and state.get('trainable_tensors') == (152 if arm == 'C' else 154)
                and state.get('trainable_numel') == (87849216 if arm == 'C' else 88233966),
                'actual pre-resume model/source/config/epoch identity: ' + arm)
        complete = arm == 'C' or not partial
        require(state.get('status') == ('complete_fixed_last20_audit' if complete else 'restorable_partial_not_completed')
                and state.get('complete') is complete, 'partial audit cannot claim completed J training')
        for key in ('strict_cpu_model_restored', 'classifier_parameters_restored_equal', 'optimizer_cpu_restored',
                    'scheduler_cpu_restored', 'scaler_cpu_restored', 'ac_cpu_restored', 'exact_ac_train_id_label_binding',
                    'python_rng_restored', 'numpy_rng_restored', 'torch_cpu_rng_restored', 'saved_cuda_rng_structure_valid',
                    'confidence_observation_binding', 'saved_state_resume_bookkeeping_valid'):
            require(state.get(key) is True, 'actual pre-resume restoration failed: ' + arm + '.' + key)
        require(state.get('optimizer_parameter_groups') == ([152] if arm == 'C' else [152, 2])
                and state.get('scheduler_epoch') == epoch and state.get('scheduler_T_max') == 20
                and state.get('ac_observation_epochs') == epoch and state.get('cuda_rng_restored_on_gpu') is False
                and state.get('classifier_head_source') == ('new_fold_H1_frozen' if arm == 'C' else 'joint_fixed_last_state')
                and state.get('classifier_equals_fresh_H1') is (arm == 'C'), 'actual pre-resume optimizer/AC/head policy: ' + arm)
    require(done.get('schema') == 'aictrack1.fresh_route.continuation.training.complete.v1'
            and done.get('complete') is True and done.get('status') == ('complete_exact_resume_to_fixed_last20' if partial else 'no_resume_needed_both_fixed_last20')
            and done.get('source_manifest_sha256') == context['training_source']['sha256']
            and done.get('completed_epoch') == 20 and done.get('completed_last_sha256') == final_sha
            and done.get('test_access') is False and done.get('new_fresh_training') is False,
            'actual complete exact resume to this same fixed-last20 required')
    require(done.get('cpu_checkpoint_audit') == items['partial_restore_audit'], 'completion must bind the actual pre-resume CPU restoration audit')
    require(done.get('terminal_cpu_checkpoint_audit') == items['terminal_restore_audit'], 'completion must bind the actual terminal CPU restoration audit')
    require(terminal.get('schema') == audit['schema'] and terminal.get('complete') is True
            and terminal.get('source_manifest_sha256') == context['training_source']['sha256']
            and terminal.get('original_diagnostic_source_manifest_sha256') == audit['original_diagnostic_source_manifest_sha256']
            and terminal.get('selected_route') == 'J5_GRAPH' and terminal.get('test_access') is False
            and terminal.get('cuda_used') is False and terminal.get('model_forwards') == 0 and terminal.get('optimizer_steps') == 0,
            'actual terminal strict two-model CPU restoration required')
    for arm in ('C', 'J5'):
        state = terminal.get(arm, {})
        wanted_sha = context['checkpoint_origins']['C']['sha256'] if arm == 'C' else final_sha
        require(state.get('checkpoint_sha256') == wanted_sha and state.get('checkpoint_epoch') == 20
                and state.get('status') == 'complete_fixed_last20_audit' and state.get('complete') is True
                and state.get('config_sha256') == context['configs'][arm]['sha256']
                and state.get('source_manifest_sha256') == context['training_source']['sha256']
                and state.get('split_sha256') == SPLIT_SHA and state.get('h1_sha256') == H1_SHA
                and state.get('trainable_tensors') == (152 if arm == 'C' else 154)
                and state.get('trainable_numel') == (87849216 if arm == 'C' else 88233966), 'actual terminal20 identity: ' + arm)
        for key in ('strict_cpu_model_restored', 'classifier_parameters_restored_equal', 'optimizer_cpu_restored',
                    'scheduler_cpu_restored', 'scaler_cpu_restored', 'ac_cpu_restored', 'exact_ac_train_id_label_binding',
                    'python_rng_restored', 'numpy_rng_restored', 'torch_cpu_rng_restored', 'saved_cuda_rng_structure_valid',
                    'confidence_observation_binding', 'saved_state_resume_bookkeeping_valid'):
            require(state.get(key) is True, 'actual terminal restoration failed: ' + arm + '.' + key)
        require(state.get('optimizer_parameter_groups') == ([152] if arm == 'C' else [152, 2])
                and state.get('scheduler_epoch') == 20 and state.get('scheduler_T_max') == 20 and state.get('ac_observation_epochs') == 20
                and state.get('cuda_rng_restored_on_gpu') is False
                and state.get('classifier_head_source') == ('new_fold_H1_frozen' if arm == 'C' else 'joint_fixed_last_state')
                and state.get('classifier_equals_fresh_H1') is (arm == 'C'), 'actual terminal20 optimizer/AC/head: ' + arm)
    for key, wanted in dict(original_phase_start_unix=ORIGINAL_START, original_deadline_unix=ORIGINAL_DEADLINE,
                            phase_start_unix=context['phase_start_unix'], deadline_unix=context['deadline_unix'],
                            initial_free_bytes=ORIGINAL_BASELINE, disk_max_bytes=DISK_MAX, min_free_bytes=MIN_FREE).items():
        require(done.get(key) == wanted, 'continuation must preserve original/shared budget: ' + key)
    require(context['phase_start_unix'] <= done.get('started_at_unix', -1)
            <= done.get('completed_at_unix', -1) < context['deadline_unix'], 'actual continuation timing outside shared window')
    attempts = done.get('actual_runtime_receipts')
    require(isinstance(attempts, list) and (bool(attempts) if partial else not attempts), 'actual continuation runtime receipts required')
    for item in attempts:
        value = actual_evidence(item, 'continuation_runtime')
        require(value.get('schema') == 'aictrack1.fresh_route.runtime.v1' and value.get('arm') == 'J5'
                and value.get('source_manifest_sha256') == context['training_source']['sha256']
                and value.get('config_sha256') == context['configs']['J5']['sha256']
                and value.get('environment') == {'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4', 'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                and value.get('actual_torch_threads') == 4 and value.get('test_access') is False,
                'actual continuation attempts must use the unchanged matched training runtime')
    if not partial:
        require(final_sha == context['checkpoint_origins']['J5']['sha256'] and done.get('owner') is None,
                'already-complete J5 must have no extra training or new trainer owner')
    return {name: items[name] for name in CHAIN_NAMES}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('paired-report', 'evidence-json', 'append-budget', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--expected-append-budget-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_sha = verify_owned_sources()
    context = validate_append_budget(args.append_budget, args.expected_append_budget_sha256)
    report, items = read(args.paired_report), read(args.evidence_json)
    validate_append_evidence(report, items, context)
    # Original 76b20 function and result schema are not modified. FULL d0c7 can
    # rebuild this exact value, including extra evidence, without monkeypatches.
    value = build_confirmation(report, items, route='J5_GRAPH', report_sha=sha(args.paired_report), source_sha=source_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, value)
    print(json.dumps(dict(complete=True, receipt=str(args.output), sha256=sha(args.output),
                         primary_delta_vs_C_pp=value['primary_delta_vs_C_pp'], test_access=False)))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

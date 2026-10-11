"""Typed, append-only time admission; never changes training, models, queries or graph ledgers."""
from __future__ import annotations
import hashlib, json, math, time
from pathlib import Path

UPSTREAM_DIAGNOSTIC_SHA = '554d7c8f839d4ab6fee1ce427c5e2b9b73435d4766ac5347794ceecace53fbc2'
BOUND_TRAINING_SOURCE_SHA = '4fc8e1420cf9347e79f7a0f607cd49a2f28735900de52477091324c06a0b8482'
TRAINING_ROOT = Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_20261009')
APPEND_ROOT = Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_retry_append_diagnostic_r1_20261009')
RETRY_CONT_ROOT = Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_retry_continuation_r1_20261009')
ORIGINAL_START = 1791497972.440027
ORIGINAL_DEADLINE = 1791526772.440027
APPEND_START = ORIGINAL_DEADLINE
APPEND_MAX_DEADLINE = 1791533972.440027
ORIGINAL_BASELINE = 13013950464
ORIGINAL_DISK_MAX = 8589934592
DISK_MAX = 12884901888
MIN_FREE = 3221225472
SCHEMA = 'aictrack1.fresh_route.continuation.retry_training_budget.v1'
from retry_recovery_contract import validate_recovery,require_common,require_disk_context
SCOPES = ['exact_J5_resume_if_needed', 'fixed_last20_C_J5_4V_diagnostic', 'fixed_J5_graph_validation']

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def actual_evidence(item, name):
    require(isinstance(item, dict) and isinstance(item.get('path'), str)
            and Path(item['path']).is_absolute() and valid_digest(item.get('sha256')),
            'actual absolute path and SHA required: ' + name)
    require(sha(item['path']) == item['sha256'], 'actual evidence drift: ' + name)
    return read(item['path'])

def validate_append_budget(path, expected_sha, *, expected_training_source_sha=BOUND_TRAINING_SOURCE_SHA,
                           live=False, now=None):
    """Metadata and actual files only. No model, image, GPU or lease is opened."""
    require(path is not None and valid_digest(expected_sha), 'explicit published appended budget and SHA required')
    path = Path(path).resolve()
    require(path.is_file() and sha(path) == expected_sha, 'published appended budget SHA mismatch')
    value = read(path)
    require(value.get('schema') == SCHEMA and value.get('status') == 'root_authorized_append_only_budget'
            and value.get('root_authorized') is True, 'typed root append-only authorization required')
    require(value.get('selected_route') == 'J5_GRAPH' and value.get('selected_arm') == 'J5'
            and value.get('test_access') is False and value.get('new_fresh_training') is False
            and value.get('scope') == SCOPES, 'only the actual preselected J5_GRAPH shared continuation and validation route')
    require(isinstance(value.get('training_root'), str) and Path(value['training_root']).resolve() == TRAINING_ROOT.resolve()
            and value.get('source_manifest_sha256') == expected_training_source_sha,
            'shared appended budget must bind the exact actual training root and source')
    require(value.get('original_phase_start_unix') == ORIGINAL_START
            and value.get('original_deadline_unix') == ORIGINAL_DEADLINE
            and value.get('initial_free_bytes') == ORIGINAL_BASELINE
            and value.get('disk_max_bytes') == DISK_MAX and value.get('min_free_bytes') == MIN_FREE,
            'original phase, cumulative allocation and baseline must never reset')
    start, deadline = value.get('phase_start_unix'), value.get('deadline_unix')
    require(start == APPEND_START and isinstance(deadline, (int, float)) and math.isfinite(deadline)
            and start < deadline <= APPEND_MAX_DEADLINE, 'shared TRAIN plus validation window exceeds original deadline plus7200s')
    original = actual_evidence(value.get('original_budget'), 'original_fresh_budget')
    require(Path(value['original_budget']['path']).resolve() == (TRAINING_ROOT / 'artifacts/queues/fresh_route_budget.json').resolve(),
            'the exact original shared training budget path is required')
    require(original == dict(schema='aictrack1.fresh_route.budget.v1', phase_start_unix=ORIGINAL_START,
                             deadline_unix=ORIGINAL_DEADLINE, initial_free_bytes=ORIGINAL_BASELINE,
                             disk_max_bytes=ORIGINAL_DISK_MAX, min_free_bytes=MIN_FREE), 'original budget must remain byte-bound and semantically unchanged')
    source = actual_evidence(value.get('training_source'), 'training_source')
    require(value['training_source']['sha256'] == expected_training_source_sha
            and Path(value['training_source']['path']).resolve() == (TRAINING_ROOT / 'source_manifest.json').resolve()
            and source.get('schema') == 'aictrack1.fresh_route.training.source.v1' and source.get('status') == 'bound'
            and source.get('selected_route') == 'J5_GRAPH' and source.get('selected_arm') == 'J5',
            'actual unchanged bound4fc8 C plus J5 source required')
    require(source.get('original_phase_start_unix') == ORIGINAL_START
            and source.get('original_deadline_unix') == ORIGINAL_DEADLINE
            and source.get('original_data_free_baseline_bytes') == ORIGINAL_BASELINE
            and source.get('disk_max_bytes') == ORIGINAL_DISK_MAX and source.get('min_free_bytes') == MIN_FREE,
            'bound producer original budget fields must remain unchanged')
    configs = value.get('configs', {})
    require(set(configs) == {'C', 'J5'}, 'exact actual C and J5 configs required')
    for arm, item in configs.items():
        _verify_file(item, 'config_' + arm)
        require(Path(item['path']).resolve() == (TRAINING_ROOT / 'configs/experiment' / ('p1_' + arm + '.yaml')).resolve()
                and source.get('files', {}).get('configs/experiment/p1_' + arm + '.yaml') == item['sha256'],
                'append budget config must be owned by actual source: ' + arm)
    checkpoints = value.get('checkpoints', {})
    require(set(checkpoints) == {'C', 'J5'}, 'actual previous fixed-last checkpoint origins required')
    for arm, item in checkpoints.items():
        require(isinstance(item, dict), 'invalid checkpoint origin: ' + arm)
        epoch = item.get('epoch')
        require(isinstance(item.get('path'), str) and valid_digest(item.get('sha256'))
                and Path(item['path']).resolve() == (TRAINING_ROOT / 'artifacts/experiments' / ('FRESH_' + arm) / 'last.pt').resolve()
                and isinstance(epoch, int) and not isinstance(epoch, bool)
                and (epoch == 20 if arm == 'C' else 1 <= epoch <= 20),
                'actual C20 and J5 partial-or20 origin metadata required')
    # J5 last.pt is legitimately replaced by exact resume. Its pre-resume SHA is
    # verified by the continuation operator's immutable CPU audit, not rehashed as
    # if the old mutable path were still the original checkpoint after completion.
    watched = value.get('watched')
    require(isinstance(watched, list) and 2 <= len(watched) <= 16, 'original controller/trainer births must be explicitly recorded')
    pids = set()
    for item in watched:
        require(isinstance(item, dict) and isinstance(item.get('role'), str) and item['role'].strip()
                and all(type(item.get(k)) is int and item[k] > 0 for k in ('pid', 'pgrp', 'session'))
                and isinstance(item.get('start_ticks'), str) and item['start_ticks'].isdigit() and int(item['start_ticks']) > 0
                and item['pid'] == item['pgrp'] == item['session'] and item['pid'] not in pids,
                'original watched process stable identity metadata invalid')
        pids.add(item['pid'])
    require(any('controller' in item['role'] for item in watched) and any('trainer' in item['role'] for item in watched),
            'both original controller and trainer sessions required')
    for pid, ticks in ((236091, '308497119'), (236098, '308497585')):
        require(any(item['pid'] == pid and item['start_ticks'] == ticks for item in watched), 'original exact C births required')
    require({'J5_controller', 'J5_trainer'} <= {item['role'] for item in watched}, 'actual J5 controller/trainer births required')
    recovery = validate_recovery(value)
    if live:
        current = time.time() if now is None else now
        require(isinstance(current, (int, float)) and math.isfinite(current) and start <= current < deadline,
                'appended shared stage has not started or has expired')
    return dict(recovery, schema='aictrack1.fresh_route.retry_append_diagnostic_context.v1',
                diagnostic_root=str(APPEND_ROOT),retry_continuation_root=str(RETRY_CONT_ROOT),
                appended_stage_budget=dict(path=str(path), sha256=expected_sha),
                original_fresh_budget=dict(value['original_budget']), training_source=dict(value['training_source']),
                configs=dict(configs), checkpoint_origins=dict(checkpoints), watched=list(watched), phase_start_unix=start, deadline_unix=deadline,
                original_phase_start_unix=ORIGINAL_START, original_deadline_unix=ORIGINAL_DEADLINE,
                original_system_baseline_bytes=ORIGINAL_BASELINE, max_cumulative_disk_bytes=DISK_MAX,
                min_actual_disk_free_bytes=MIN_FREE, selected_route='J5_GRAPH', selected_arm='J5',
                restored_checkpoint_origin='same_actual_bound_training_source_fixed_last20',
                old_deadline_rewritten=False, training_budget_reset=False, graph_ledger_reset=False, test_access=False)

def _verify_file(item, name):
    require(isinstance(item, dict) and isinstance(item.get('path'), str) and Path(item['path']).is_absolute()
            and valid_digest(item.get('sha256')) and sha(item['path']) == item['sha256'], 'actual evidence drift: ' + name)

def require_deadline(context, requested, *, now=None):
    current = time.time() if now is None else now
    require(isinstance(requested, (int, float)) and math.isfinite(requested)
            and context['phase_start_unix'] <= current < requested <= context['deadline_unix'],
            'entry deadline must fit the one published shared append window')

def require_pair_context(candidate, control, context):
    for arm, item in [('J5', candidate), ('C', control)]:
        identity = item['identity']
        require(identity.get('model_arm') == arm and identity.get('checkpoint_role') == 'last'
                and identity.get('checkpoint_epoch') == 20 and identity.get('selected_route') == 'J5_GRAPH',
                'actual complete selected fixed-last20 C/J5 required')
        require(identity.get('append_stage_context') == context,
                'both actual restored caches must bind this same immutable appended stage')
        require(identity.get('training_source_manifest_sha256') == context['training_source']['sha256']
                and identity.get('experiment_config_sha256') == context['configs'][arm]['sha256'],
                'actual restored source/config origin must match appended budget')

def require_exact_groups_closed(watched, *, proc=Path('/proc')):
    """Bounded read-only process identity check; no signaling or lock mutation."""
    roots = [path for path in Path(proc).iterdir() if path.name.isdigit()]
    require(len(roots) <= 16384, 'bounded process inventory exceeded')
    groups, sessions = {item['pgrp'] for item in watched}, {item['session'] for item in watched}
    births = {item['pid']: item for item in watched}
    for root in roots:
        try:
            raw = (root / 'stat').read_text(encoding='ascii')
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(len(raw) <= 8192, 'oversize process stat')
        fields = raw.rsplit(') ', 1)[1].split()
        require(len(fields) >= 20, 'short process stat')
        value = dict(pid=int(raw.split(' (', 1)[0]), pgrp=int(fields[2]), session=int(fields[3]), start_ticks=fields[19])
        if value['pid'] in births:
            require(all(value[key] == births[value['pid']][key] for key in value), 'prior PID birth or ownership changed')
            raise RuntimeError('prior exact controller/trainer still alive')
        require(value['pgrp'] not in groups and value['session'] not in sessions,
                'prior exact session/group still contains a process')
    return True

def execution_admission(context, evidence_path):
    from write_append_confirmation import verify_continuation_chain
    require(evidence_path is not None, 'real continuation execution chain required before GPU admission')
    items = read(evidence_path)
    done = actual_evidence(items.get('continuation_completion'), 'continuation_completion')
    final_sha = done.get('completed_last_sha256')
    require(valid_digest(final_sha), 'actual completed J5 fixed-last SHA required')
    chain = verify_continuation_chain(items, context, final_sha)
    for arm, digest in (('C', context['checkpoint_origins']['C']['sha256']), ('J5', final_sha)):
        checkpoint = TRAINING_ROOT / 'artifacts/experiments' / ('FRESH_' + arm) / 'last.pt'
        require(sha(checkpoint) == digest, 'actual final checkpoint drift before frozen GPU entry: ' + arm)
    watched = list(context['watched'])
    owner = done.get('owner')
    if owner is not None:
        require(isinstance(owner, dict) and all(type(owner.get(key)) is int and owner[key] > 0 for key in ('pid','pgrp','session'))
                and owner['pid'] == owner['pgrp'] == owner['session'] and owner['pid'] not in {item['pid'] for item in watched}
                and isinstance(owner.get('start_ticks'), str) and owner['start_ticks'].isdigit(), 'exact new trainer closure identity required')
        watched.append(dict(owner, role='continuation_trainer'))
    require_exact_groups_closed(watched)
    return chain

def annotate_audit_context(output, context, chain):
    """Metadata-only annotation after unchanged strict restoration and four-view inference."""
    output = Path(output)
    runtime_path, audit_path, report_path = [output / name for name in ('runtime_receipt.json', 'p1_checkpoint_audit.json', 'report.json')]
    runtime, audit, report = read(runtime_path), read(audit_path), read(report_path)
    require(audit.get('complete') is True and audit.get('checkpoint_role') == 'last' and audit.get('epoch') == 20
            and audit.get('classifier_parameters_restored_equal') is True
            and audit.get('exact_ac_train_id_label_binding') is True, 'append metadata cannot bless an incomplete actual model restoration')
    require(time.time() < context['deadline_unix'], 'complete raw validation exceeded shared appended stage deadline')
    completed = time.time()
    runtime['append_stage_context'] = context
    runtime['continuation_execution_chain'] = chain
    runtime['appended_stage_completed_at_unix'] = completed
    _write(runtime_path, runtime)
    audit.update(append_stage_context=context, appended_stage_completed_at_unix=completed,
                 continuation_execution_chain=chain, runtime_receipt_sha256=sha(runtime_path))
    _write(audit_path, audit)
    report['append_stage_context'] = context
    report['appended_stage_completed_at_unix'] = completed
    _write(report_path, report)

def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    temporary.replace(path)

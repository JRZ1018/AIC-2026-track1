"""CPU-only train-held-out linear-head signal audit; not a formal candidate.

Original project implementation, recipe/provenance frozen by docs217. Runtime
must be copied to a new independent guarded scope, never run in a dirty checkout.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--config', required=True)
    config_path = Path(args.parse_args().config).resolve()
    root = config_path.parent
    assert root == Path('/root/AICtrack1_heldout_supervision_audit_20261007')
    cfg = json.loads(config_path.read_text())
    stage = json.loads((root/'stage_receipt.json').read_text())
    manifest_path = root/'sources.json'
    assert sha(manifest_path) == os.environ['HELDOUT_SOURCE_SHA']
    for name, digest in json.loads(manifest_path.read_text())['files'].items():
        assert '..' not in Path(name).parts and not Path(name).is_absolute()
        assert sha(root/name) == digest
    sys.path[:0] = [str(root/'src'), str(root/'scripts')]
    from mainline_guard import memory_snapshot, memory_reason, LOCK, process_identity
    # The canonical queue attaches the trainer just after Popen. Fail closed,
    # allowing only this bounded startup race, before loading any tensors.
    for attempt in range(50):
        lock = json.loads(LOCK.read_text())
        if (lock['owner'] == cfg['owner'] and lock['trainer_pid'] == os.getpid()
                and lock['queue_pid'] == os.getppid() and lock['config'] == str(config_path)):
            break
        assert attempt < 49, 'canonical lease identity not attached'
        time.sleep(.1)
    anchor = process_identity(os.getpid())
    token = lock['token']
    assert cfg['fold_seed'] == 20261007 and cfg['folds'] == 3 and cfg['num_classes'] == 750
    assert memory_reason(memory_snapshot(), starting=True) is None

    def guard():
        assert time.time() < stage['deadline_epoch'], 'actual audit deadline'
        free = shutil.disk_usage(root).free
        assert free > 7*1024**3 and stage['initial_free_bytes']-free < 512*1024**2
        assert memory_reason(memory_snapshot()) is None
        live = json.loads(LOCK.read_text())
        assert live['token'] == token and live['trainer_pid'] == os.getpid()
        assert live['queue_pid'] == os.getppid() and process_identity(os.getpid()) == anchor

    guard()
    import torch
    import torch.nn.functional as F
    from aictrack1.data.feature_cache import load_feature_cache
    from aictrack1.data.contracts import load_dataset_config
    from aictrack1.data.indexed_images import read_indexed_images
    from aictrack1.engine.frozen_head import FrozenHeadConfig, FrozenHeadTrainer
    from aictrack1.models.heads import build_classifier_head
    from aictrack1.methods.confidence import ClasswiseHistoryWeighting, classwise_rank_noise_scores
    from aictrack1.methods.heldout_supervision import grouped_folds, exclusion_contract
    from mainline_h1_contract import bind_rows, validate_cache_metadata, validate_final, HEAD_CONFIG, MANIFEST_SHA, SPLIT_SHA
    torch.set_num_threads(2)
    prior = Path(cfg['prior_root'])
    assert sha(prior/'mainline_sources_sha256.json') == cfg['prior_source_sha256']
    assert sha(prior/'inputs/split.csv') == SPLIT_SHA
    raw_manifest = Path(cfg['manifest'])
    assert sha(raw_manifest) == MANIFEST_SHA
    cache = load_feature_cache(prior/'artifacts/features/mainline336')
    validate_cache_metadata(cache.metadata.__dict__)
    records = read_indexed_images(load_dataset_config(prior/'configs/data/mainline336.yaml'),
                                 source='train', manifest_path=raw_manifest,
                                 split_path=prior/'inputs/split.csv')
    positions, binding = bind_rows(cache.sample_ids, cache.labels.tolist(), cache.partitions, records)
    assert len(positions) == 118958
    assert cache.relative_paths == tuple(r.relative_path for r in records)
    assert cache.sample_indices.tolist() == [r.sample_index for r in records]
    with raw_manifest.open(encoding='utf-8-sig') as handle:
        raw = {r['sample_id']: r for r in csv.DictReader(handle)}
    rows = [dict(raw[cache.sample_ids[p]], partition='train') for p in positions]
    assert all(cache.partitions[p] == 'train' for p in positions)
    assignments, groups = grouped_folds(rows, folds=3, seed=20261007)
    plans = [exclusion_contract(rows, assignments, groups, heldout_fold=k, num_classes=750) for k in range(3)]
    x = cache.features[positions].float()
    y = cache.labels[positions].long()
    ids = cache.sample_indices[positions]
    assert x.shape == (118958, 512) and torch.isfinite(x).all()
    head_path = prior/'artifacts/experiments/mainline-h1/last.pt'
    assert sha(head_path) == cfg['head_sha256']
    original = torch.load(head_path, weights_only=True, map_location='cpu')
    validate_final(original, binding)
    control_path = Path(cfg['control_last'])
    assert sha(control_path) == cfg['control_last_sha256']
    control = torch.load(control_path, weights_only=True, map_location='cpu', mmap=True)
    ac = ClasswiseHistoryWeighting(ids, y, momentum=.8, history_mix=.5, min_weight=.02, magnitude_mix=.75)
    ac.load_state_dict(control['confidence_weighting'])
    assert ac.epochs_updated == 20
    ac_weights = ac.weights_for(ids)
    del control, cache
    output = root/'artifacts/experiments/heldout-audit'
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.yaml').write_bytes(config_path.read_bytes())
    (output/'command.txt').write_text(json.dumps(sys.argv))
    fits = []
    scores = {}

    def score(model, select):
        result = {key: [] for key in ('ce', 'py', 'argmax')}
        with torch.inference_mode():
            for start in range(0, len(select), 4096):
                guard(); pos = select[start:start+4096]
                logits = model(x[pos])
                assert torch.isfinite(logits).all()
                ce = F.cross_entropy(logits, y[pos], reduction='none')
                result['ce'].append(ce); result['py'].append((-ce).exp())
                result['argmax'].append(logits.argmax(1))
        return {key: torch.cat(v) for key, v in result.items()}

    original_head = build_classifier_head('linear', 512, 750)
    original_head.load_state_dict(original['model'])
    scores['original_h1'] = score(original_head, list(range(len(y))))
    del original, original_head
    scores['heldout'] = dict(ce=torch.empty(len(y)), py=torch.empty(len(y)), argmax=torch.empty(len(y), dtype=torch.long))
    coverage = torch.zeros(len(y), dtype=torch.long)
    def exact_state(a, b):
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
        elif isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys()
            for key in a: exact_state(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert type(a) is type(b) and len(a) == len(b)
            for x0, x1 in zip(a, b): exact_state(x0, x1)
        else:
            assert a == b
    for number in range(4):
        fit, held, contract = (list(range(len(y))), list(range(len(y))), {'role':'all_train_cpu_reference'}) if number == 0 else plans[number-1]
        guard()
        # Fresh initial head every time; no learned H1 initialization or held-out labels.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261005)
            trainer = FrozenHeadTrainer(build_classifier_head('linear', 512, 750),
                                        FrozenHeadConfig(**HEAD_CONFIG), device='cpu', num_classes=750)
        fit_x, fit_y = x[fit], y[fit]
        for epoch in range(100):
            guard()
            loss = trainer.train_epoch(fit_x, fit_y)
            assert math.isfinite(loss)
            trainer.history.append(dict(epoch=epoch+1, train_loss=loss,
                                        learning_rate=float(trainer.optimizer.param_groups[0]['lr'])))
            if (epoch+1) % 10 == 0:
                print(json.dumps(dict(head=number, epoch=epoch+1, loss=loss, elapsed=time.time()-stage['started_epoch'])), flush=True)
        assert trainer.epoch == trainer.scheduler.last_epoch == 100 and len(trainer.optimizer.state) == 2
        # Manual CPU-only serialization avoids the generic trainer's CUDA RNG query.
        state = dict(model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                     scheduler=trainer.scheduler.state_dict(), scaler=trainer.scaler.state_dict(),
                     config=HEAD_CONFIG, history=trainer.history, contract=contract,
                     role='audit_only_no_teacher_no_fusion', epoch=100)
        assert shutil.disk_usage(root).free-stage['min_free_bytes'] > 64*1024**2
        assert stage['initial_free_bytes']-shutil.disk_usage(root).free+64*1024**2 < 512*1024**2
        temporary = output/f'head{number}.pt.tmp'
        torch.save(state, temporary)
        safe = torch.load(temporary, weights_only=True, map_location='cpu')
        restored = FrozenHeadTrainer(build_classifier_head('linear',512,750), FrozenHeadConfig(**HEAD_CONFIG),device='cpu',num_classes=750)
        restored.model.load_state_dict(safe['model']);restored.optimizer.load_state_dict(safe['optimizer'])
        restored.scheduler.load_state_dict(safe['scheduler']);restored.scaler.load_state_dict(safe['scaler'])
        assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in safe['model'].items())
        assert all(torch.isfinite(v).all() for slot in restored.optimizer.state.values() for v in slot.values() if isinstance(v,torch.Tensor))
        for key, value in (('optimizer', restored.optimizer.state_dict()),
                           ('scheduler', restored.scheduler.state_dict()),
                           ('scaler', restored.scaler.state_dict())):
            exact_state(state[key], safe[key]); exact_state(safe[key], value)
        assert safe['epoch'] == 100 and safe['history'] == trainer.history and safe['contract'] == contract
        temporary.rename(output/f'head{number}.pt')
        values = score(trainer.model,held)
        if number == 0:scores['all_train_cpu'] = values
        else:
            for key,v in values.items():scores['heldout'][key][held] = v
            coverage[held] += 1
        fits.append(dict(head=number, fit=contract, optimizer_steps=sum(math.ceil(len(fit)/4096) for _ in range(100)),sha256=sha(output/f'head{number}.pt')))
        del trainer,restored,state,safe,fit_x,fit_y
    assert (coverage == 1).all(), 'each train row must be held out exactly once'
    assert all(torch.isfinite(v).all() for item in scores.values() for v in item.values())
    ranks = {k:classwise_rank_noise_scores(v['ce'],y) for k,v in scores.items()}
    ac_rank = classwise_rank_noise_scores(-ac_weights,y)
    summary = []
    for c in range(750):
        m=y==c
        summary.append(dict(class_id=c,count=int(m.sum()), ac_mass=float(ac_weights[m].sum()),
                            heldout_agreement=float((scores['heldout']['argmax'][m]==y[m]).float().mean()),
                            original_agreement=float((scores['original_h1']['argmax'][m]==y[m]).float().mean()),
                            rank_move_gt_point2=float((abs(ranks['heldout'][m]-ranks['all_train_cpu'][m])>=.2).float().mean())))
    def corr(a,b):
        value=torch.corrcoef(torch.stack([a,b]))[0,1]
        return float(value) if torch.isfinite(value) else None
    counts = torch.bincount(y, minlength=750)
    tail_classes = sorted(range(750), key=lambda c: (int(counts[c]), c))[:150]
    tail_mask = torch.zeros(len(y), dtype=torch.bool)
    for c in tail_classes: tail_mask |= y == c
    hard_mask = scores['original_h1']['argmax'] != y
    slices = {}
    for name, mask in (('all', torch.ones(len(y), dtype=torch.bool)),
                       ('tail150_by_train_count', tail_mask), ('original_h1_disagreement', hard_mask),
                       ('tail_and_original_h1_disagreement', tail_mask & hard_mask)):
        assert mask.any()
        slices[name] = dict(count=int(mask.sum()), original_ac_mass=float(ac_weights[mask].sum()),
                           original_ac_mean=float(ac_weights[mask].mean()),
                           heldout_agreement=float((scores['heldout']['argmax'][mask] == y[mask]).float().mean()),
                           rank_move_vs_cpu_ref_ge_point2=float((abs(ranks['heldout'][mask]-ranks['all_train_cpu'][mask]) >= .2).float().mean()),
                           rank_move_vs_ac20_ge_point2=float((abs(ranks['heldout'][mask]-ac_rank[mask]) >= .2).float().mean()))
    report=dict(complete=True,train_rows=118958,classes=750,group_count=len(set(groups)),
                folds=[p[2] for p in plans],fits=fits,class_quality=summary,slices=slices,
                tail_definition='150 smallest train class counts; class index tie break, descriptive only',
                row_binding=binding,source_sha256=sha(manifest_path),head_sha256=cfg['head_sha256'],
                config_sha256=sha(config_path),control_last_sha256=cfg['control_last_sha256'],
                heldout_vs_original_rank_corr=corr(ranks['heldout'],ranks['original_h1']),
                heldout_vs_cpu_ref_rank_corr=corr(ranks['heldout'],ranks['all_train_cpu']),
                heldout_vs_original_ac20_rank_corr=corr(ranks['heldout'],ac_rank),
                original_h1_vs_cpu_ref_argmax_equal=float((scores['original_h1']['argmax']==scores['all_train_cpu']['argmax']).float().mean()),
                formal_candidate_launched=False, noise_precision_claim=False,
                validation_used_for_scoring_or_fit=False,test_access=False,gpu_access=False,
                finished_epoch=time.time(),limitations=['FP16 fixed center cache','no clean label truth','CPU vs original CUDA fitting path','held-out heads fit fewer rows and batches','one fold partition seed; not training variance','no automatic new curriculum from disagreement'])
    guard()
    score_payload = dict(scores=scores, sample_ids=[r['sample_id'] for r in rows],sample_indices=ids,
                         targets=y,folds=torch.tensor(assignments),groups=groups,ac20_weights=ac_weights,
                         row_binding=binding,config_sha256=sha(config_path),source_sha256=sha(manifest_path))
    score_tmp = output/'scores.pt.tmp'
    torch.save(score_payload, score_tmp)
    exact_state(score_payload, torch.load(score_tmp, weights_only=True, map_location='cpu'))
    score_tmp.rename(output/'scores.pt')
    report['scores_sha256'] = sha(output/'scores.pt')
    (root/'audit_result.json').write_text(json.dumps(report,indent=2))
    (output/'metrics.json').write_text(json.dumps(dict(selected_checkpoint='head0.pt',role='audit_reference_only_not_final_model',complete=True,test_access=False)))
    print(json.dumps({k:v for k,v in report.items() if k not in ('class_quality','folds')}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

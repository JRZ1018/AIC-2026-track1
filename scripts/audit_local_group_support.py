"""Original bounded, train-only local component support audit (docs219).

No fitted heads, optimizer, image forward, validation scoring, test, teacher or
label repair. Reuse official initial CLIP336 cache only, not adapted features.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    config=Path(parser.parse_args().config).resolve(); root=config.parent
    assert root==Path('/root/AICtrack1_local_group_support_audit_20261007')
    cfg=json.loads(config.read_text()); stage=json.loads((root/'stage_receipt.json').read_text())
    assert sha(root/'sources.json')==os.environ['LOCAL_GROUP_SOURCE_SHA']
    for name,digest in json.loads((root/'sources.json').read_text())['files'].items():
        assert '..' not in Path(name).parts and not Path(name).is_absolute()
        assert sha(root/name)==digest
    sys.path[:0]=[str(root/'src'),str(root/'scripts')]
    from mainline_guard import LOCK, memory_snapshot, memory_reason, process_identity
    for attempt in range(50):
        lease=json.loads(LOCK.read_text())
        if lease['owner']==cfg['owner'] and lease['trainer_pid']==os.getpid() and lease['queue_pid']==os.getppid() and lease['config']==str(config):
            break
        assert attempt<49; time.sleep(.1)
    token=lease['token']; anchor=process_identity(os.getpid())
    def guard():
        assert time.time()<stage['deadline_epoch']
        free=shutil.disk_usage(root).free
        assert free>stage['min_free_bytes'] and stage['initial_free_bytes']-free<stage['disk_peak_limit_bytes']
        assert memory_reason(memory_snapshot()) is None
        live=json.loads(LOCK.read_text())
        assert live['token']==token and live['trainer_pid']==os.getpid() and live['queue_pid']==os.getppid()
        assert process_identity(os.getpid())==anchor
    guard()
    import torch
    from aictrack1.data.feature_cache import load_feature_cache
    from aictrack1.data.contracts import load_dataset_config
    from aictrack1.data.indexed_images import read_indexed_images
    from aictrack1.methods.heldout_supervision import grouped_folds
    from aictrack1.methods.local_group_support import local_group_support
    from mainline_h1_contract import bind_rows, validate_cache_metadata, MANIFEST_SHA, SPLIT_SHA
    torch.set_num_threads(2)
    assert cfg['num_classes']==750 and cfg['k']==16 and cfg['query_batch']==8 and cfg['sample_seed']==20261003
    prior=Path(cfg['prior_root']); evidence=Path(cfg['heldout_root'])
    assert sha(prior/'mainline_sources_sha256.json')==cfg['prior_source_sha256']
    assert sha(prior/'inputs/split.csv')==SPLIT_SHA and sha(cfg['manifest'])==MANIFEST_SHA
    accepted_report=json.loads((evidence/'audit_result.json').read_text())
    assert sha(evidence/'audit_result.json')==cfg['heldout_report_sha256']
    assert json.loads((evidence/'controller_result.json').read_text())['complete']
    scalars=evidence/'artifacts/experiments/heldout-audit/scores.pt'
    assert sha(scalars)==accepted_report['scores_sha256']==cfg['heldout_scores_sha256']
    v=torch.load(scalars,map_location='cpu',weights_only=True)
    cache=load_feature_cache(prior/'artifacts/features/mainline336')
    validate_cache_metadata(cache.metadata.__dict__)
    records=read_indexed_images(load_dataset_config(prior/'configs/data/mainline336.yaml'),source='train',manifest_path=Path(cfg['manifest']),split_path=prior/'inputs/split.csv')
    positions,binding=bind_rows(cache.sample_ids,cache.labels.tolist(),cache.partitions,records)
    assert len(positions)==118958 and cache.relative_paths==tuple(r.relative_path for r in records)
    assert cache.sample_indices.tolist()==[r.sample_index for r in records]
    with Path(cfg['manifest']).open(encoding='utf-8-sig',newline='') as f:
        raw={r['sample_id']:r for r in csv.DictReader(f)}
    rows=[dict(raw[cache.sample_ids[p]],partition='train') for p in positions]
    _,groups=grouped_folds(rows,folds=3,seed=20261007)
    x=cache.features[positions].float(); y=cache.labels[positions].long()
    assert x.shape==(118958,512) and torch.isfinite(x).all()
    assert v['sample_ids']==[r['sample_id'] for r in rows] and torch.equal(v['targets'],y) and v['groups']==groups
    assert v['row_binding']==binding
    del cache
    queries=[]
    for c in range(750):
        candidates=(y==c).nonzero().flatten().tolist(); assert candidates
        queries.append(min(candidates,key=lambda i:hashlib.sha256(f"20261003:{rows[i]['sample_id']}".encode()).digest()))
    guard(); started=time.time()
    result=local_group_support(x,y,groups,queries,num_classes=750,k=16,query_batch=8,guard=guard)
    elapsed=time.time()-started
    q=torch.tensor(queries); support=result['target_support']; ac=v['ac20_weights'][q]
    counts=torch.bincount(y,minlength=750); tail=torch.isin(y[q],torch.tensor(sorted(range(750),key=lambda c:(int(counts[c]),c))[:150]))
    hard=v['scores']['original_h1']['argmax'][q]!=y[q]
    summary={}
    for name,mask in [('all',torch.ones(750,dtype=torch.bool)),('tail150',tail),('h1_disagreement',hard),('tail_and_h1_disagreement',tail&hard)]:
        summary[name]=dict(count=int(mask.sum()),mean_target_support=float(support[mask].mean()) if mask.any() else None,
                          positive_target_support=int((support[mask]>0).sum()),
                          original_ac_mean=float(ac[mask].mean()) if mask.any() else None)
    corr=lambda z:float(torch.corrcoef(torch.stack([support,z]))[0,1]) if support.std()>0 and z.std()>0 else None
    names=result['component_names']; neighbours=[[names[g] for g in neighbours] for neighbours in result['neighbour_components']]
    assert all(groups[i] not in neighbours[j] and len(set(neighbours[j]))==16 for j,i in enumerate(queries))
    output=root/'artifacts/experiments/local-group-support'; output.mkdir(parents=True,exist_ok=False)
    payload=dict(sample_ids=[rows[i]['sample_id'] for i in queries],query_row_indices=q,targets=y[q],
                 groups=[groups[i] for i in queries],neighbour_groups=neighbours,target_support=support,
                 cosines=result['cosines'],ac20_weights=ac,source_sha256=sha(root/'sources.json'),
                 config_sha256=sha(config),row_binding=binding,k=16)
    torch.save(payload,output/'query_support.pt'); guard()
    report=dict(complete=True,finished_epoch=time.time(),train_rows=118958,queries=750,classes=750,
                query_id_sha256=hashlib.sha256('\n'.join(payload['sample_ids']).encode()).hexdigest(),
                group_count=len(names),k=16,group_exclusion_zero_leakage=True,distinct_component_votes=True,
                optimizer_steps=0,head_fit_steps=0,gpu_access=False,test_access=False,validation_used=False,
                formal_candidate_launched=False,noise_precision_claim=False,summary=summary,
                raw_support_vs_ac20_corr=corr(ac),
                raw_support_vs_h1_py_corr=corr(v['scores']['original_h1']['py'][q]),
                raw_support_vs_heldout_py_corr=corr(v['scores']['heldout']['py'][q]),
                support_zero_queries=int((support==0).sum()),support_one_queries=int((support==1).sum()),
                computation_seconds=elapsed,query_support_sha256=sha(output/'query_support.pt'),
                source_sha256=sha(root/'sources.json'),config_sha256=sha(config),row_binding=binding,
                limitations=['one fixed query per observed class; not full-train classwise ranking',
                             'neighbour labels are noisy; support not noise precision',
                             'raw cross-class correlation confounded by class difficulty',
                             'fixed k16 is descriptive, no coefficient/k scanning',
                             'FP16 initial CLIP center cache; visual semantics may mismatch fine-grained labels',
                             'no new weighting policy or candidate authorized by mechanical audit PASS'])
    with (root/'audit_result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

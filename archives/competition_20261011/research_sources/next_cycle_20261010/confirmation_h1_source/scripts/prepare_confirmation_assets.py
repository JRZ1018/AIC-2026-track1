"""CPU-only split and immutable official-feature audit; never trains or decodes images."""
from __future__ import annotations
import argparse, csv, dataclasses, hashlib, io, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aictrack1.data.manifest import read_manifest
from aictrack1.data.splits import build_grouped_stratified_split, leakage_group
MANIFEST_SHA='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
OLD_SPLIT_SHA='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c'
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest_rows(rows):
    return hashlib.sha256(json.dumps(rows,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def write_new(path, payload):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    raw=payload if isinstance(payload,bytes) else (json.dumps(payload,indent=2,sort_keys=True)+'\n').encode()
    if p.exists():
        if p.read_bytes()!=raw:raise ValueError('immutable output differs: '+str(p))
    else:p.write_bytes(raw)
def split_csv(assignment):
    s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(['sample_id','split'])
    w.writerows(sorted(assignment.items()));return s.getvalue().encode()
def read_split(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    d={r['sample_id']:r['split'] for r in rows}
    if len(d)!=len(rows) or set(d.values())-{'train','validation'}:raise ValueError('invalid prior split')
    return d
def connected_groups(records):
    """Union exact hash, exact-group and near-group relations before holdout exclusion."""
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    first={}
    for i,r in enumerate(records):
        for kind,value in [('sha256',r.sha256),('exact',r.duplicate_group),('near',r.near_duplicate_group)]:
            if value:
                key=(kind,value)
                if key in first:union(i,first[key])
                else:first[key]=i
    components=defaultdict(list)
    for i in range(len(records)):components[find(i)].append(i)
    names={};changed=0
    for indices in components.values():
        original={leakage_group(records[i]) for i in indices}
        group=next(iter(original)) if len(original)==1 else 'union-'+min(records[i].sample_id for i in indices)
        changed+=len(original)>1
        for i in indices:names[records[i].sample_id]=group
    return names,dict(component_count=len(components),merged_priority_groups=changed,
                     non_singleton_components=sum(len(x)>1 for x in components.values()))
def build_split(records, old):
    if set(old)!={r.sample_id for r in records}:raise ValueError('prior split manifest mismatch')
    groups,connection=connected_groups(records)
    forbidden={groups[sid] for sid,part in old.items() if part=='validation'}
    derived=[dataclasses.replace(r,near_duplicate_group=groups[r.sample_id]) for r in records]
    assignment=build_grouped_stratified_split(derived,validation_fraction=.2,seed=20261009,
        min_train_per_class=1,group_order='seeded_random',forbidden_validation_groups=frozenset(forbidden))
    audit_split(records,assignment,groups,forbidden)
    return assignment,groups,connection,forbidden
def audit_split(records, assignment, groups, forbidden):
    seen=defaultdict(set)
    for r in records:
        part=assignment[r.sample_id];seen[groups[r.sample_id]].add(part)
        if part=='validation' and groups[r.sample_id] in forbidden:raise ValueError('old selection query reintroduced')
    if any(len(x)!=1 for x in seen.values()):raise ValueError('connected group crosses train/query')
def audit_feature_cache(path, records, assignment):
    import torch
    torch.set_num_threads(2)
    root=Path(path);meta=json.loads((root/'metadata.json').read_text());complete=json.loads((root/'complete.json').read_text())
    expected=dict(source='train',dataset_stage='rematch',manifest_sha256=MANIFEST_SHA,
        class_mapping_sha256=MAPPING_SHA,clip_weight_sha256=CLIP_SHA,feature_dim=512,
        normalized=True,expected_record_count=148695,peft_checkpoint_sha256=None,peft_config_sha256=None)
    for k,v in expected.items():
        if meta.get(k)!=v:raise ValueError('non-official/frozen feature metadata '+k)
    pre=meta.get('preprocessing',{})
    if pre.get('resize')!=336 or pre.get('center_crop')!=336 or pre.get('position_interpolation')!='bicubic_7x7_to_10x10':raise ValueError('not official336 transform')
    names=[x['name'] for x in complete['shards']]
    if names!=[p.name for p in sorted((root/'shards').glob('part-*.pt'))] or len(names)!=37:raise ValueError('shard sequence mismatch')
    offset=0;shards=[];row_digest=[];samples=[]
    for entry in complete['shards']:
        p=root/'shards'/entry['name']
        if sha(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise ValueError('feature shard SHA/size drift '+p.name)
        data=torch.load(p,map_location='cpu',weights_only=True,mmap=True);f=data['features'];n=len(f)
        if f.shape!=(n,512) or f.dtype!=torch.float16 or not torch.isfinite(f).all():raise ValueError('feature shape/dtype/finite mismatch')
        norms=torch.linalg.vector_norm(f.float(),dim=1)
        if not torch.allclose(norms,torch.ones_like(norms),atol=2e-3,rtol=2e-3):raise ValueError('feature normalization mismatch')
        for j,r in enumerate(records[offset:offset+n]):
            if data['sample_ids'][j]!=r.sample_id or int(data['labels'][j])!=r.internal_class_index or int(data['sample_indices'][j])!=offset+j or data['relative_paths'][j]!=r.relative_path:raise ValueError('actual official feature row binding mismatch')
            row_digest.append([r.sample_id,r.internal_class_index,assignment[r.sample_id]])
        samples.append(dict(shard=p.name,sample_id=data['sample_ids'][0],target=int(data['labels'][0]),feature_finite=True))
        shards.append(entry);offset+=n
    if offset!=len(records) or complete['record_count']!=offset:raise ValueError('feature coverage missing')
    train=[r for r in row_digest if r[2]=='train']
    binding=dict(schema='aictrack1.mainline.h1.rows.v1',counts=dict(Counter(x[2] for x in row_digest)),
        all_rows_sha256=digest_rows(row_digest),train_rows_sha256=digest_rows(train),validation_rows_sha256=digest_rows([r for r in row_digest if r[2]=='validation']),
        validation_used_for_optimization_or_selection=False)
    return dict(original_root=str(root),metadata_sha256=sha(root/'metadata.json'),complete_sha256=sha(root/'complete.json'),
        original_metadata=meta,record_count=offset,shards=shards,row_binding=binding,all_shard_rows_verified=True,
        row_fields_verified=['sample_id','labels','sample_indices','relative_paths'],sample_checks=samples,
        rebind_method='original immutable feature tensors; override only partitions/metadata after full positive row binding',
        fresh_feature_extraction=False,test_access=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--prior-split',type=Path,required=True)
    p.add_argument('--feature-cache',type=Path,required=True);p.add_argument('--class-mapping',type=Path,required=True);p.add_argument('--label-conflicts',type=Path,required=True)
    a=p.parse_args();start=time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    if sha(a.manifest)!=MANIFEST_SHA or sha(a.prior_split)!=OLD_SPLIT_SHA:raise ValueError('official/prior input SHA mismatch')
    records=read_manifest(a.manifest)
    if len(records)!=148695 or any(r.decode_status!='ok' or r.source_split!='train' for r in records):raise ValueError('Stage2 only/all decodable manifest required')
    with a.class_mapping.open(encoding='utf-8-sig',newline='') as f:mapping=list(csv.DictReader(f))
    if len(mapping)!=750 or any(mapping[r.internal_class_index]['official_class_id']!=r.official_class_id or int(mapping[r.internal_class_index]['internal_class_index'])!=r.internal_class_index for r in records):raise ValueError('class ID mapping differs')
    old=read_split(a.prior_split);assignment,groups,connection,forbidden=build_split(records,old)
    counts=Counter(assignment.values());supports=[]
    for target in range(750):
        rows=[r for r in records if r.internal_class_index==target]
        expected=min(round(len(rows)*.2),len(rows)-1);actual=sum(assignment[r.sample_id]=='validation' for r in rows)
        available=sum(groups[r.sample_id] not in forbidden for r in rows)
        supports.append(dict(target=target,official_class_id=mapping[target]['official_class_id'],total=len(rows),train=len(rows)-actual,validation=actual,target_validation=expected,available_remaining_rows=available,validation_deficit=expected-actual))
    split_raw=split_csv(assignment);split_sha=hashlib.sha256(split_raw).hexdigest();write_new(ROOT/'inputs/confirmation_split.csv',split_raw)
    features=audit_feature_cache(a.feature_cache,records,assignment)
    derived=dict(features['original_metadata']);derived.update(split_sha256=split_sha)
    write_new(ROOT/'inputs/rebound_feature_metadata.json',derived)
    write_new(ROOT/'inputs/feature_identity.json',features)
    rows=io.StringIO(newline='');w=csv.writer(rows,lineterminator='\n');w.writerow(['sample_id','target','split','group','relative_path'])
    w.writerows((r.sample_id,r.internal_class_index,assignment[r.sample_id],groups[r.sample_id],r.relative_path) for r in records)
    write_new(ROOT/'inputs/confirmation_rows.csv',rows.getvalue().encode())
    audit=dict(schema='aictrack1.confirmation.assets.v1',status='split_features_verified_h1_not_yet_trained',competition_stage='stage2',
        split_seed=20261009,training_seed=20261008,group_order='seeded_random',counts=dict(counts),num_classes=750,
        train_query_connected_group_overlap=0,old_selection_query_confirmation_group_overlap=0,old_selection_query_rows_overlap=0,
        groups=connection,forbidden_validation_group_count=len(forbidden),support_deficits=[x for x in supports if x['validation_deficit']],per_class_support=supports,
        h1_row_binding=features['row_binding'],h1_selection_rule='fixed epoch100 last; new-fold train only; no validation selection',
        test_access=False,gpu_access=False,assets={name:dict(local_path=str(path),sha256=sha(path)) for name,path in dict(manifest=a.manifest,prior_split=a.prior_split,class_mapping=a.class_mapping,label_conflicts=a.label_conflicts,split=ROOT/'inputs/confirmation_split.csv',rows=ROOT/'inputs/confirmation_rows.csv',feature_identity=ROOT/'inputs/feature_identity.json',rebound_feature_metadata=ROOT/'inputs/rebound_feature_metadata.json').items()})
    write_new(ROOT/'inputs/confirmation_assets.json',audit)
    print(json.dumps(dict(status=audit['status'],split_sha256=split_sha,counts=dict(counts),support_deficit_classes=len(audit['support_deficits']),connected_components=connection,seconds=time.monotonic()-start)))
if __name__=='__main__':main()

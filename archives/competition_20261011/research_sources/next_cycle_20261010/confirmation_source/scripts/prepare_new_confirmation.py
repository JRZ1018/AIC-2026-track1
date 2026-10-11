"""CPU-only pre-result confirmation split; exclude both past query component unions."""
from __future__ import annotations
import argparse,csv,dataclasses,hashlib,io,json,os,sys,time
from collections import Counter,defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aictrack1.data.manifest import read_manifest
from aictrack1.data.splits import build_grouped_stratified_split,leakage_group
MANIFEST_SHA='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
MAPPING_SHA='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
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
    parser=argparse.ArgumentParser();parser.add_argument('--feature-cache',type=Path)
    a=parser.parse_args();started=time.monotonic();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    pool_path=ROOT/'inputs/QUERY_POOL_AUDIT.json';pool=json.loads(pool_path.read_text(encoding='utf-8'))
    if sha(pool_path)!='ca0eb2f76e4a24c57538065f1b890d40ae01c41637e210392fa52b0fb2f43497':raise ValueError('actual pre-result pool audit drift')
    sources=pool['sources']
    for k,v in sources.items():
        if sha(v['path'])!=v['sha256']:raise ValueError('actual source drift '+k)
    records=read_manifest(sources['manifest']['path']);ids={r.sample_id for r in records}
    if len(records)!=148695 or len(ids)!=len(records) or any(r.decode_status!='ok' or r.source_split!='train' for r in records):raise ValueError('all official train rows required')
    groups,connection=connected_groups(records)
    old=read_split(sources['old_query_split']['path']);used=read_split(sources['used_confirmation_split']['path'])
    if set(old)!=ids or set(used)!=ids:raise ValueError('old split ID coverage differs')
    old_groups={groups[s] for s,v in old.items() if v=='validation'};used_groups={groups[s] for s,v in used.items() if v=='validation'};forbidden=old_groups|used_groups
    eligible=[r for r in records if groups[r.sample_id] not in forbidden]
    if len(forbidden)!=57318 or len(eligible)!=89221 or len({groups[r.sample_id] for r in eligible})!=82678:raise ValueError('independent eligible pool integer mismatch')
    eligible_counts=Counter(r.internal_class_index for r in eligible)
    eligible_groups=defaultdict(set)
    for r in eligible:eligible_groups[r.internal_class_index].add(groups[r.sample_id])
    for item in pool['eligible_per_class']:
        if eligible_counts[item['target']]!=item['rows'] or len(eligible_groups[item['target']])!=item['connected_groups']:raise ValueError('eligible support differs from prior pool audit')
    derived=[dataclasses.replace(r,near_duplicate_group=groups[r.sample_id]) for r in records]
    assignment=build_grouped_stratified_split(derived,validation_fraction=.2,seed=20261010,min_train_per_class=1,group_order='seeded_random',forbidden_validation_groups=frozenset(forbidden))
    audit_split(records,assignment,groups,forbidden)
    counts=Counter(assignment.values());querygroups={groups[r.sample_id] for r in records if assignment[r.sample_id]=='validation'};traingroups={groups[r.sample_id] for r in records if assignment[r.sample_id]=='train'}
    if querygroups&traingroups or querygroups&old_groups or querygroups&used_groups:raise ValueError('group leakage')
    byclass=defaultdict(list)
    for r in records:byclass[r.internal_class_index].append(r)
    mapping_path=Path('D:/aicomp/AICtrack1/artifacts/audits/stage2/metadata/classes.csv')
    if sha(mapping_path)!='0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be':raise ValueError('official mapping file differs')
    with mapping_path.open(encoding='utf-8-sig',newline='') as f:mapping=list(csv.DictReader(f))
    supports=[]
    for target in range(750):
        rows=byclass[target];q=[r for r in rows if assignment[r.sample_id]=='validation'];t=[r for r in rows if assignment[r.sample_id]=='train']
        if not q or not t:raise ValueError('class cannot supply positive train/query: '+str(target))
        if any(mapping[target]['official_class_id']!=r.official_class_id for r in rows):raise ValueError('mapping row alignment')
        desired=min(round(len(rows)*.2),len(rows)-1)
        supports.append(dict(target=target,official_class_id=mapping[target]['official_class_id'],total=len(rows),train=len(t),validation=len(q),validation_connected_groups=len({groups[r.sample_id] for r in q}),target_validation=desired,available_remaining_rows=eligible_counts[target],available_remaining_connected_groups=len(eligible_groups[target]),validation_deficit=desired-len(q)))
    split_raw=split_csv(assignment);split_sha=hashlib.sha256(split_raw).hexdigest()
    featureprior=json.loads((ROOT/'inputs/official_feature_identity_prior.json').read_text(encoding='utf-8'));cache=a.feature_cache or Path(featureprior['original_root'])
    features=audit_feature_cache(cache,records,assignment)
    if features['metadata_sha256']!=featureprior['metadata_sha256'] or features['complete_sha256']!=featureprior['complete_sha256'] or features['shards']!=featureprior['shards']:raise ValueError('original immutable37 cache identity differs')
    metadata=dict(features['original_metadata']);metadata['split_sha256']=split_sha
    rows=io.StringIO(newline='');writer=csv.writer(rows,lineterminator='\n');writer.writerow(['sample_id','target','split','group','relative_path']);writer.writerows((r.sample_id,r.internal_class_index,assignment[r.sample_id],groups[r.sample_id],r.relative_path) for r in records)
    write_new(ROOT/'inputs/confirmation_split.csv',split_raw);write_new(ROOT/'inputs/confirmation_rows.csv',rows.getvalue().encode());write_new(ROOT/'inputs/row_binding.json',features['row_binding']);write_new(ROOT/'inputs/feature_identity.json',features);write_new(ROOT/'inputs/rebound_feature_metadata.json',metadata)
    binding=features['row_binding'];expected_rows=[[r.sample_id,r.internal_class_index,assignment[r.sample_id]] for r in records]
    if binding['all_rows_sha256']!=digest_rows(expected_rows) or binding['counts']!=dict(counts):raise ValueError('split row binding mismatch')
    report=dict(schema='aictrack1.next_cycle10.confirmation.assets.v1',status='actual_split_features_verified_H1_not_trained',split_seed=20261010,training_seed=20261008,validation_fraction=.2,group_order='seeded_random',min_train_per_class=1,expected_total=148695,counts=dict(counts),num_classes=750,train_class_count=750,query_class_count=750,target_query_approximately=29737,actual_query_unique_rows=counts['validation'],actual_query_connected_groups=len(querygroups),train_query_connected_group_overlap=0,old0c9_query_connected_group_overlap=0,used57189_query_connected_group_overlap=0,old_query_row_overlap=sum(old[s]=='validation' for s,v in assignment.items() if v=='validation'),used_query_row_overlap=sum(used[s]=='validation' for s,v in assignment.items() if v=='validation'),forbidden_validation_group_count=len(forbidden),eligible_unused_query_rows=len(eligible),connected_components=connection,minimum_query_rows=min(x['validation'] for x in supports),minimum_query_independent_groups=min(x['validation_connected_groups'] for x in supports),supported20_classes=sum(x['validation_connected_groups']>=20 for x in supports),support_deficits=[x for x in supports if x['validation_deficit']],per_class_support=supports,split_sha256=split_sha,h1_row_binding=binding,future_H1_checkpoint_sha256=None,new_H1_trained=False,new_visual_training_started=False,visual_control='fresh joint J5 alpha.1 fixedlast20',visual_selected_candidate=None,selection_policy='one exploration winner only; identical officialCLIP+newtrainH1 initialization; fixed20/P7/fixedgraph',test_access=False,gpu_access=False,images_decoded=False,alpha_validation_results_read=False,used_test_or_online_scores=False,source_audit_sha256=sha(pool_path),source_refs=sources,official37_features=features['original_root'],source_assets={'split':{'path':str(ROOT/'inputs/confirmation_split.csv'),'sha256':split_sha},'rows':{'path':str(ROOT/'inputs/confirmation_rows.csv'),'sha256':sha(ROOT/'inputs/confirmation_rows.csv')},'feature_identity':{'path':str(ROOT/'inputs/feature_identity.json'),'sha256':sha(ROOT/'inputs/feature_identity.json')},'rebound_feature_metadata':{'path':str(ROOT/'inputs/rebound_feature_metadata.json'),'sha256':sha(ROOT/'inputs/rebound_feature_metadata.json')}},actual_CPU_runtime_seconds=time.monotonic()-started)
    write_new(ROOT/'inputs/confirmation_assets.json',report)
    print(json.dumps({k:report[k] for k in ['status','counts','split_sha256','actual_query_connected_groups','minimum_query_rows','minimum_query_independent_groups','supported20_classes','train_query_connected_group_overlap','old0c9_query_connected_group_overlap','used57189_query_connected_group_overlap','actual_CPU_runtime_seconds']},indent=2))
if __name__=='__main__':main()

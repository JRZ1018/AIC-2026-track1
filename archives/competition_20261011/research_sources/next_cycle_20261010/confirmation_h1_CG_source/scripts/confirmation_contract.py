"""Positive new-fold identities and original canonical-lease checks."""
from __future__ import annotations
import csv, hashlib, json, os, shutil, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEPLOYMENT_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_confirmation_h1_CG_20261010')
QUERY_SOURCE_SHA='51a79ffbaf94fbb333cdab49123e55ccc3fb5cf2fe78cfdb6b5ea86da8810e50'
SPLIT_SHA='610111665e1055bcda7157add15b16ade353138365e54ee87dc92316e6dfd6f1'
LOCK='/root/autodl-tmp/AICtrack1/artifacts/queues/training.lock.json'
HEAD_CONFIG=dict(epochs=100,batch_size=4096,learning_rate=.01,weight_decay=0.,momentum=.9,
                 gradient_clip_norm=None,seed=20261008,optimizer='adam',scheduler='constant',amp=False)
SCOPE='fold_train_fixed_final_epochs'
SELECTION='fixed epoch100 last; new-fold train only; no validation selection'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_new(path,payload):
    raw=(json.dumps(payload,indent=2,sort_keys=True)+'\n').encode();p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()!=raw:raise ValueError('immutable file differs '+str(p))
    if not p.exists():p.write_bytes(raw)
def require(value,message):
    if not value:raise ValueError(message)

def verify_source(require_bound=True):
    d=read_json(ROOT/'source_manifest.json')
    require(d.get('schema')=='aictrack1.next_cycle10.CG_confirmation.h1.source.v1','unexpected H1 source schema')
    require(d.get('status') in {'unbound_template','bound_h1_only'},'unknown H1 source status')
    if require_bound:require(d['status']=='bound_h1_only','H1 source not bound for GPU')
    for rel,expected in d['files'].items():
        p=Path(rel)
        require(not p.is_absolute() and '..' not in p.parts and (ROOT/p).resolve().is_relative_to(ROOT),'source path escape')
        require(sha(ROOT/p)==expected,'frozen H1 source drift '+rel)
    definition=read_json(ROOT/'inputs/query_definition_source_manifest.json')
    require(sha(ROOT/'inputs/query_definition_source_manifest.json')==QUERY_SOURCE_SHA,'wrong pre-result query definition')
    for rel,expected in definition['files'].items():
        if rel.startswith('inputs/') and (ROOT/rel).is_file():require(sha(ROOT/rel)==expected,'frozen query input drift '+rel)
    return d

def normalized_assets():
    a=read_json(ROOT/'inputs/confirmation_assets.json')
    require(a['schema']=='aictrack1.next_cycle10.confirmation.assets.v1' and a['split_sha256']==SPLIT_SHA,'wrong new-fold assets')
    require(a['counts']=={'train':118958,'validation':29737} and a['train_class_count']==750 and a['query_class_count']==750,'new fold count/support differs')
    require(a['old0c9_query_connected_group_overlap']==a['used57189_query_connected_group_overlap']==a['train_query_connected_group_overlap']==0,'old query or train leakage')
    require(a['future_H1_checkpoint_sha256'] is None and not a['new_H1_trained'] and not a['test_access'],'pre-result definition differs')
    value=dict(a);value['assets']={'manifest':a['source_refs']['manifest'],'split':{'sha256':SPLIT_SHA}}
    return value

def validate_budget(b,now=None):
    from CG_admission_contract import validate_confirmation_budget
    return validate_confirmation_budget(b,now)

def validate_admission(a):
    from CG_admission_contract import validate_CG_admission
    return validate_CG_admission(a)

def audit_class_mapping(path):
    from aictrack1.data.class_mapping import ClassMapping
    expected_bytes='0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be'
    expected_logical='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'
    require(sha(path)==expected_bytes,'actual classes.csv byte SHA differs')
    mapping=ClassMapping.from_csv(path)
    require(len(mapping)==750 and mapping.digest()==expected_logical,'actual logical mapping identity differs')
    return dict(file_sha256=expected_bytes,logical_mapping_sha256=expected_logical,num_classes=len(mapping))

def audit_external_cache(cache):
    identity=read_json(ROOT/'inputs/feature_identity.json');cache=Path(cache)
    require(sha(cache/'metadata.json')==identity['metadata_sha256'] and sha(cache/'complete.json')==identity['complete_sha256'],'immutable official feature identity mismatch')
    require(len(identity['shards'])==37,'all official37 shards required')
    for item in identity['shards']:
        p=cache/'shards'/item['name'];require(sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'actual feature shard drift '+item['name'])
    return identity

def deployment():
    require(os.name=='posix' and ROOT.resolve()==DEPLOYMENT_ROOT.resolve(),'exact dedicated Linux H1 ROOT required')
    d=read_json(ROOT/'inputs/h1_deployment.json');a=normalized_assets()
    require(d.get('schema')=='aictrack1.next_cycle10.confirmation.h1.deployment.v1' and d.get('canonical_lock')==LOCK and d.get('assets_receipt_sha256')==sha(ROOT/'inputs/confirmation_assets.json'),'H1 deployment binding differs')
    require(d['runtime']=={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_num_threads':4},'runtime mismatch')
    for name,item in d['assets'].items():require(sha(item['path'])==item['sha256'],'actual deployed asset drift '+name)
    require(d.get('class_mapping_identity')==audit_class_mapping(d['assets']['class_mapping']['path']),'deployed file/logical class mapping identity differs')
    require(d['budget_sha256']==sha(ROOT/'inputs/budget.json') and d['admission_sha256']==sha(ROOT/'inputs/root_admission.json'),'actual budget/admission binding differs')
    validate_budget(read_json(ROOT/'inputs/budget.json'),time.time());validate_admission(read_json(ROOT/'inputs/root_admission.json'))
    audit_external_cache(d['feature_cache'])
    return d,a

def runtime_check():
    import torch
    for k,v in {'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}.items():
        if os.environ.get(k)!=v:raise RuntimeError('launch environment drift '+k)
    torch.set_num_threads(4)
    if torch.get_num_threads()!=4:raise RuntimeError('actual torch thread count differs')
def resource_check(reserve=0):
    free=shutil.disk_usage(ROOT).free;initial=int(os.environ.get('P1_INITIAL_FREE','0'))
    if (time.time()>=float(os.environ.get('P1_DEADLINE','0')) or initial<=0 or free-reserve<int(os.environ.get('P1_MIN_FREE',str(3*1024**3))) or initial-free+reserve>int(os.environ.get('P1_DISK_MAX',str(20*1024**3)))):raise RuntimeError('confirmation deadline/disk exceeded')
def entry(config,resume):
    if os.name=='nt':raise RuntimeError('actual H1 invocation is root-owned remote Linux only')
    verify_source();d,a=deployment();runtime_check()
    if sha(ROOT/'source_manifest.json')!=os.environ.get('P1_SOURCE_SHA'):raise RuntimeError('H1 source environment binding mismatch')
    from aictrack1.engine.training_queue import read_training_lock
    relative=Path(config).resolve().relative_to(ROOT).as_posix()
    for attempt in range(50):
        r=read_training_lock(d['canonical_lock'])
        if r and r.token==os.environ.get('P1_LEASE_TOKEN') and r.trainer_pid==os.getpid() and r.queue_pid==os.getppid() and r.config==relative:break
        if attempt==49:raise RuntimeError('H1 exact canonical lease mismatch')
        time.sleep(.1)
    if relative!='configs/experiment/confirmation_H1.yaml':raise ValueError('exact H1 config required')
    if resume is not None and Path(resume).resolve()!=ROOT/'artifacts/experiments/confirmation_H1/last.pt':raise ValueError('resume wrong H1 state')
    import yaml
    experiment=yaml.safe_load(Path(config).read_text(encoding='utf-8'))['experiment']
    expected=dict(run_id='20261010-confirmation-H1',execution_enabled=True,training_scope=SCOPE,training=HEAD_CONFIG,output_dir='../../artifacts/experiments/confirmation_H1',device='cuda')
    require(experiment==expected,'exact actual H1 YAML/config differs')
    b=read_json(ROOT/'inputs/budget.json')
    for key,value in [('P1_DEADLINE',b['deadline_unix']),('P1_INITIAL_FREE',b['adjusted_initial_free_bytes']),('P1_DISK_MAX',b['disk_max_bytes']),('P1_MIN_FREE',b['min_free_bytes'])]:require(float(os.environ.get(key,'0'))==float(value),'H1 launch budget environment differs '+key)
    resource_check();return d,a
def row_binding(rows):
    from prepare_confirmation_assets import digest_rows
    typed=[[r['sample_id'],int(r['target']),r['split']] for r in rows]
    return dict(schema='aictrack1.mainline.h1.rows.v1',counts={p:sum(x[2]==p for x in typed) for p in ('train','validation')},
        all_rows_sha256=digest_rows(typed),train_rows_sha256=digest_rows([x for x in typed if x[2]=='train']),
        validation_rows_sha256=digest_rows([x for x in typed if x[2]=='validation']),validation_used_for_optimization_or_selection=False)
def load_rebound_cache(d,a):
    from dataclasses import replace
    import torch
    from aictrack1.data.feature_cache import load_feature_cache
    cache=load_feature_cache(d['feature_cache']);identity=read_json(ROOT/'inputs/feature_identity.json')
    if cache.metadata.__dict__!=identity['original_metadata'] or cache.metadata.peft_checkpoint_sha256 is not None:raise ValueError('only exact frozen official cache can rebind')
    with (ROOT/'inputs/confirmation_rows.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    # Positive official manifest equality, not a checksum-only declaration.
    with Path(d['assets']['manifest']['path']).open(encoding='utf-8-sig',newline='') as f:official=list(csv.DictReader(f))
    if len(rows)!=148695 or len(official)!=len(rows):raise ValueError('official rows missing')
    for i,(r,o) in enumerate(zip(rows,official,strict=True)):
        if (o['source_split']!='train' or r['sample_id']!=o['sample_id'] or r['relative_path']!=o['relative_path'] or int(r['target'])!=int(o['internal_class_index']) or cache.sample_ids[i]!=r['sample_id'] or int(cache.labels[i])!=int(r['target']) or int(cache.sample_indices[i])!=i or cache.relative_paths[i]!=r['relative_path']):raise ValueError('official feature/new-fold row binding differs')
    binding=row_binding(rows)
    if binding!=a['h1_row_binding']:raise ValueError('H1 positive row digest differs')
    metadata=replace(cache.metadata,split_sha256=a['assets']['split']['sha256'])
    rebound=replace(cache,metadata=metadata,partitions=tuple(r['split'] for r in rows))
    if not torch.isfinite(rebound.features).all():raise ValueError('nonfinite rebound cache')
    if not torch.allclose(torch.linalg.vector_norm(rebound.features.float(),dim=1),torch.ones(len(rows)),atol=2e-3,rtol=2e-3):raise ValueError('official feature norm differs')
    return rebound,binding
def validate_h1(state,a,binding):
    if (state.get('training_scope')!=SCOPE or state.get('epoch')!=100 or state.get('config')!=HEAD_CONFIG or state.get('selection_rule')!=SELECTION or state.get('head')!={'kind':'linear'} or state.get('row_binding')!=binding):raise ValueError('H1 fixed100/config/positive row binding differs')
    meta=state.get('feature_cache_metadata',{})
    if meta.get('training_scope')!=SCOPE or meta.get('training_record_count')!=a['counts']['train'] or meta.get('split_sha256')!=a['assets']['split']['sha256']:raise ValueError('H1 train-only metadata mismatch')
    if state.get('best_model') is not None or [x.get('epoch') for x in state.get('history',[])]!=list(range(1,101)) or any(set(x)!={'epoch','train_loss','learning_rate'} for x in state['history']):raise ValueError('H1 validation/best selection or incomplete history')
    required={'optimizer','scheduler','scaler','python_rng_state','numpy_rng_state','torch_rng_state','cuda_rng_state_all','loss_config','class_weights','best_top1'}
    if not required.issubset(state) or state['scheduler'].get('last_epoch')!=100:raise ValueError('H1 complete optimizer/scheduler/scaler/RNG state required')
    groups=state['optimizer'].get('param_groups',[])
    if len(groups)!=1 or len(groups[0].get('params',[]))!=2 or len(state['optimizer'].get('state',{}))!=2:raise ValueError('H1 two-parameter Adam state required')
    if set(state['model'])!={'linear.weight','linear.bias'}:raise ValueError('single linear head required')
    import torch
    if any(not isinstance(state['model'][k],torch.Tensor) or state['model'][k].shape!=shape or not torch.isfinite(state['model'][k]).all() for k,shape in [('linear.weight',(750,512)),('linear.bias',(750,))]):raise ValueError('H1 exact finite linear tensor shape required')
    for key,expected in [('competition_stage','stage2'),('train_manifest_sha256',a['assets']['manifest']['sha256']),('split_sha256',a['assets']['split']['sha256']),('class_mapping_sha256','811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879'),('clip_weight_sha256','40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af')]:
        if state.get(key)!=expected:raise ValueError('positive H1 provenance mismatch '+key)

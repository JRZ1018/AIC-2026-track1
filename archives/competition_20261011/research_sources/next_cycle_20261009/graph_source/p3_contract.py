"""Strict heldout-cache identities and precomputed unique queries; no model load."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CLIP_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
FEATURE_PROTOCOL='L2_normalize(mean(L2_normalize(raw_visual_each_of_four_views))) in float32 on CPU'
VIEW_PROTOCOL='native336 center and exact flip plus resize_short_side384_center_crop336 and exact flip; one checkpoint; four float logits equally averaged'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def order_sha(ids):return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def verify_source():
    manifest=read(ROOT/'source_manifest.json')
    if manifest.get('schema')!='aictrack1.p3.graph_source.v1':raise ValueError('graph source identity missing')
    for name,digest in manifest['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not (ROOT/p).resolve().is_relative_to(ROOT):raise ValueError('source escape')
        if sha(ROOT/p)!=digest:raise ValueError('graph source drift: '+name)
    return manifest

def evidence(path):
    p=Path(path).resolve()
    return dict(path=str(p),sha256=sha(p))

def validate_feature_payload(feature,logits,identity):
    """Pure metadata/order checker; labels are not supplied to graph functions."""
    import numpy as np
    import torch
    if (feature.get('schema')!='aictrack1.stage2_train_derived_4v_features.v1'
            or feature.get('feature_protocol')!=FEATURE_PROTOCOL or feature.get('checkpoint_role')!='last'
            or feature.get('checkpoint_epoch')!=20 or feature.get('additional_model_forwards')!=0
            or feature.get('test_access') is not False or feature.get('observed_layout')!='four_calls_B'
            or feature.get('view_order')!=['native','native_exact_flip','scaled384_crop336','scaled_exact_flip']):
        raise ValueError('only the same frozen checkpoint four existing forwards may supply features')
    for name in ('checkpoint_sha256','experiment_config_sha256','split_sha256','manifest_sha256','class_mapping_sha256','view_protocol'):
        if feature.get(name)!=logits.get(name) or feature.get(name)!=identity.get(name):raise ValueError('feature/logit identity differs: '+name)
    values=feature['features']
    if (not isinstance(values,torch.Tensor) or values.dtype!=torch.float32 or values.ndim!=2 or values.shape[1]!=512
            or len(values)!=len(logits['logits']) or not torch.isfinite(values).all()):raise ValueError('finite float32 [N,512] features required')
    norms=torch.linalg.vector_norm(values,dim=1);nonzero=norms>0
    if not torch.allclose(norms[nonzero],torch.ones_like(norms[nonzero]),atol=2e-5,rtol=0):raise ValueError('features are not already normalized')
    ids=list(feature['sample_ids'])
    if ids!=list(logits['sample_ids']) or len(set(ids))!=len(ids) or order_sha(ids)!=identity.get('sample_ids_order_sha256'):
        raise ValueError('feature/logit row order or unique IDs differ')
    if not torch.equal(feature['targets'].long(),logits['targets'].long()):raise ValueError('cached row labels differ')
    if feature['view_protocol']!=VIEW_PROTOCOL:raise ValueError('the frozen four-view protocol is required')
    return values.numpy(),ids

def load_assets(path,expected_sha):
    import numpy as np
    import torch,yaml
    if sha(path)!=expected_sha:raise ValueError('graph assets receipt SHA differs')
    value=read(path)
    if (value.get('schema')!='aictrack1.p3.graph_assets.v1' or value.get('status')!='verified_train_derived_same_checkpoint_4v'
            or value.get('test_access') is not False or value.get('graph_protocol_sha256')!=sha(ROOT/'graph_protocol.json')):
        raise ValueError('registered heldout graph assets required')
    ev=value['evidence']
    for name,item in ev.items():
        if not Path(item['path']).is_absolute() or sha(item['path'])!=item['sha256']:raise ValueError('graph asset drift: '+name)
    identity=read(ev['identity']['path'])
    if (identity.get('schema')!='aictrack1.p0_frozen_fold_input.v1' or identity.get('stage')!=2
            or identity.get('holdout_verified') is not True or identity.get('test_access') is not False
            or identity.get('training_scope')!='fixed_split_validation' or identity.get('checkpoint_role')!='last'
            or identity.get('checkpoint_epoch')!=20 or identity.get('training_count')!=118958
            or identity.get('validation_count')!=29737 or identity.get('num_classes')!=750
            or identity.get('training_validation_overlap_rows')!=0 or identity.get('train_validation_exact_group_overlap')!=0
            or identity.get('train_validation_near_group_overlap')!=0):raise ValueError('legal complete heldout fixed-last20 required')
    bindings={'checkpoint':'checkpoint_sha256','config':'experiment_config_sha256','features':'feature_file_sha256',
        'logits':'bundle_sha256','rows':'row_csv_sha256','checkpoint_audit':'source_receipt_sha256',
        'training_source':'training_source_manifest_sha256','diagnostic_source':'diagnostic_source_manifest_sha256',
        'split':'split_sha256','h1':'h1_sha256'}
    for name,field in bindings.items():
        if ev[name]['sha256']!=identity.get(field):raise ValueError('cache source binding differs: '+name)
    source=read(ev['training_source']['path'])
    if ev['config']['sha256'] not in source['files'].values() or ev['model_config']['sha256'] not in source['files'].values():
        raise ValueError('actual config/model missing from frozen source')
    model=yaml.safe_load(Path(ev['model_config']['path']).read_text(encoding='utf-8'))['model']
    if model.get('name')!='ViT-B/32' or model.get('expected_weight_sha256')!=CLIP_SHA or model.get('runtime_downloads_allowed') is not False:
        raise ValueError('only the specified official CLIP backbone may provide cached features')
    raw=yaml.safe_load(Path(ev['config']['path']).read_text(encoding='utf-8'))['experiment']
    if raw.get('training_scope')!='fixed_split_validation' or raw['training']['epochs']!=20:raise ValueError('cache was not trained as legal fixed20 holdout')
    audit=read(ev['checkpoint_audit']['path'])
    if (audit.get('schema') not in ('aictrack1.p1.fixed_last_diagnostic_audit.v1','aictrack1.confirmation.fixed_last_diagnostic_audit.v1')
            or audit.get('fixed_last_sha256')!=identity['checkpoint_sha256'] or audit.get('checkpoint_role')!='last'
            or audit.get('checkpoint_epoch')!=20 or audit.get('test_access') is not False
            or audit.get('classifier_parameters_restored_equal') is not True or audit.get('model_frozen_eval') is not True
            or audit.get('inference_optimizer_steps')!=0
            or audit.get('training_source_manifest_sha256')!=ev['training_source']['sha256']
            or audit.get('diagnostic_source_manifest_sha256')!=ev['diagnostic_source']['sha256']):raise ValueError('actual fixed-last frozen model audit differs')
    logits=torch.load(ev['logits']['path'],map_location='cpu',weights_only=True,mmap=True)
    feature=torch.load(ev['features']['path'],map_location='cpu',weights_only=True,mmap=True)
    if (logits.get('schema')!='aictrack1.stage2_train_derived_fused_logits.v1'
            or tuple(logits['logits'].shape)!=(29737,750) or not torch.isfinite(logits['logits']).all()):raise ValueError('complete finite 4V logits required')
    features,ids=validate_feature_payload(feature,logits,identity)
    targets=logits['targets'].long().numpy();counts=logits['train_class_counts'].long().numpy()
    if counts.shape!=(750,) or counts.sum()!=118958 or (counts<=0).any():raise ValueError('train support binding differs')
    with Path(ev['rows']['path']).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if ([r['sample_id'] for r in rows]!=ids or not np.array_equal([int(r['target']) for r in rows],targets)):
        raise ValueError('verified source rows differ')
    groups=[r['group'] for r in rows]
    if not all(groups):raise ValueError('explicit duplicate-group identities required')
    query=read(ev['query_identity']['path'])
    if (query.get('schema') not in ('aictrack1.p1.frozen_queries.v1','aictrack1.confirmation.frozen_queries.v1')
            or query.get('control_ids_sha256')!=order_sha(ids) or query.get('selection_uses_predictions') is not False
            or query.get('test_access') is not False or query.get('query_indices_sha256')!=ev['query_indices']['sha256']):
        raise ValueError('frozen support-only unique query identity differs')
    if query.get('schema')=='aictrack1.confirmation.frozen_queries.v1' and query.get('split_sha256')!=identity['split_sha256']:
        raise ValueError('confirmation queries belong to another split')
    with np.load(ev['query_indices']['path'],allow_pickle=False) as z:arrays={key:z[key].copy() for key in z.files}
    protocols=[]
    for name,prefix in [('natural_unique_images','natural'),('balanced_supported_unique','balanced')]:
        indices=arrays[prefix+'_indices'];columns=arrays[prefix+'_columns']
        if (indices.ndim!=1 or columns.ndim!=1 or indices.dtype.kind not in 'iu' or columns.dtype.kind not in 'iu'
                or len(indices)==0 or len(columns)<2 or len(set(indices.tolist()))!=len(indices)
                or np.any(indices<0) or np.any(indices>=len(ids)) or np.any(columns<0) or np.any(columns>=750)
                or not np.array_equal(columns,np.unique(columns))):raise ValueError('query rows/closed classes are not unique and ordered')
        selected_targets=targets[indices];local=np.searchsorted(columns,selected_targets)
        if np.any(local>=len(columns)) or not np.array_equal(columns[local],selected_targets):raise ValueError('query label is outside frozen closed class set')
        p=next(x for x in query['protocols'] if x['name']==name)
        if ev[prefix+'_queries']['sha256']!=p['query_csv_sha256']:raise ValueError('query CSV SHA differs')
        with Path(ev[prefix+'_queries']['path']).open(encoding='utf-8-sig',newline='') as f:qr=list(csv.DictReader(f))
        if (len(qr)!=len(indices) or [r['sample_id'] for r in qr]!=[ids[int(i)] for i in indices]
                or [int(r['source_row']) for r in qr]!=indices.tolist()
                or [int(r['target']) for r in qr]!=selected_targets.tolist()
                or [r['group'] for r in qr]!=[groups[int(i)] for i in indices]):raise ValueError('frozen query subset does not map to source rows')
        if prefix=='natural' and (not np.array_equal(indices,np.arange(len(ids))) or not np.array_equal(columns,np.arange(750))):
            raise ValueError('natural protocol must cover all rows/classes once')
        if prefix=='balanced' and (len(set(groups[int(i)] for i in indices))!=len(indices)
                or not np.all(np.bincount(local,minlength=len(columns))==20)):
            raise ValueError('balanced protocol must have 20 unique groups per supported class')
        protocols.append(dict(name=name,indices=indices,columns=columns,targets=local,
            groups=[groups[int(i)] for i in indices],ids=[ids[int(i)] for i in indices],train_counts=counts[columns]))
    mismatches=int(np.sum(logits['logits'].float().argmax(1).numpy()!=np.asarray([int(r['four_view_prediction']) for r in rows])))
    return dict(assets=value,identity=identity,features=features,scores=logits['logits'].float().numpy(),protocols=protocols,
        precision=dict(cached_logits_dtype=str(logits['logits'].dtype),feature_dtype=str(feature['features'].dtype),
            computation_dtype='float32',fp16_argmax_mismatches_vs_original_fp32=mismatches,
            note='float32 computation cannot recover any precision discarded by the saved logits'),ids=ids)

def bind(arguments):
    import yaml
    identity=read(arguments.identity);base=Path(arguments.identity).resolve().parent
    resolve=lambda value:Path(value) if Path(value).is_absolute() else base/value
    config=Path(arguments.config).resolve();raw=yaml.safe_load(config.read_text(encoding='utf-8'))['experiment']
    query=Path(arguments.query_directory).resolve()
    input_path=lambda value:(config.parent/Path(value)).resolve()
    files=dict(features=arguments.features,logits=arguments.logits,identity=arguments.identity,
        checkpoint=arguments.checkpoint,config=config,model_config=(config.parent/raw['model_config']).resolve(),
        split=input_path(raw['split']),h1=input_path(raw['head_checkpoint']),
        training_source=arguments.training_source,diagnostic_source=arguments.diagnostic_source,
        rows=resolve(identity['row_csv']),checkpoint_audit=resolve(identity['source_provenance_receipt']),
        query_identity=query/'query_identity.json',query_indices=query/'query_indices.npz',
        natural_queries=query/'natural_unique_images_queries.csv',balanced_queries=query/'balanced_supported_unique_queries.csv')
    value=dict(schema='aictrack1.p3.graph_assets.v1',status='verified_train_derived_same_checkpoint_4v',
        graph_protocol_sha256=sha(ROOT/'graph_protocol.json'),evidence={k:evidence(p) for k,p in files.items()},
        test_access=False,parameters_use_test_or_online_scores=False)
    output=Path(arguments.output);output.parent.mkdir(parents=True,exist_ok=True)
    # Validation first uses an owned temporary input, never claims real assets from placeholders.
    temporary=output.with_name(output.name+'.validation.tmp')
    with temporary.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
    try:loaded=load_assets(temporary,sha(temporary))
    finally:temporary.unlink()
    with output.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
    return dict(status='graph_assets_verified',path=str(output.resolve()),sha256=sha(output),
        checkpoint_sha256=loaded['identity']['checkpoint_sha256'],precision=loaded['precision'])

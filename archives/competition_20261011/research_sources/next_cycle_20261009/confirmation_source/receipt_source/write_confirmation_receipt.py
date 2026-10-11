"""Independent CPU receipt writer after real fresh C/J diagnostics; no model training/inference."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
SPLIT='57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7'
CLIP='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
MANIFEST='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d'
REQUIRED=['checkpoint_C','config_C','identity_C','checkpoint_audit_C','checkpoint_J','config_J','identity_J','checkpoint_audit_J','paired_report','training_source','diagnostic_source','selection_protocol','frozen_postprocess','split','h1','h1_receipt','runtime_C','runtime_J','root_review']
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def evidence_files(request):
    result={}
    for name in REQUIRED:
        item=request.get('evidence',{}).get(name);path=item.get('path') if isinstance(item,dict) else item
        require(isinstance(path,str) and Path(path).is_absolute() and Path(path).is_file(),'actual absolute evidence required '+name)
        digest=sha(path)
        if isinstance(item,dict) and 'sha256' in item:require(digest==item['sha256'],'supplied SHA differs '+name)
        result[name]=dict(path=str(Path(path).resolve()),sha256=digest)
    return result
def verify_source(item):
    source=read(item['path']);root=Path(item['path']).parent
    require(isinstance(source.get('files'),dict) and source['files'],'positive source inventory missing')
    for rel,expected in source['files'].items():
        p=Path(rel);require(not p.is_absolute() and '..' not in p.parts and (root/p).resolve().is_relative_to(root),'source escape')
        digest=expected.get('sha256') if isinstance(expected,dict) else expected
        require(sha(root/p)==digest,'actual source drift '+rel)
    return source,root
def validate_runtime(C,J,ev,arm):
    for name,value,label in [('C',C,'C'),('J',J,arm)]:
        require(value.get('schema')=='aictrack1.confirmation.runtime.v1' and value.get('arm')==label and value.get('test_access') is False,'actual runtime receipt missing/wrong arm')
        for field,evidence_name in [('source_sha256','training_source'),('config_sha256','config_'+name),('split_sha256','split'),('h1_sha256','h1')]:require(value.get(field)==ev[evidence_name]['sha256'],'runtime source/config/split/H1 binding differs '+field)
        r=value.get('runtime',{})
        for k,v in dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',torch_num_threads=4).items():require(r.get(k)==v,'actual runtime mismatch '+k)
    require(C['runtime'].get('torch_num_interop_threads')==J['runtime'].get('torch_num_interop_threads') and isinstance(C['runtime'].get('torch_num_interop_threads'),int),'C/J actual interop threads differ/missing')
    return dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',actual_torch_threads=4,equal_between_arms=True,
                torch_num_interop_threads=C['runtime']['torch_num_interop_threads'])
def validate_report(report,ev,arm):
    require(report.get('schema')=='aictrack1.confirmation.fixed_last_paired.v1' and report.get('complete') is True and report.get('selected_arm')==arm and report.get('p1_arm')==arm,'fresh paired report not complete/selected arm differs')
    require(report.get('test_access') is False and report.get('used_test_or_online_scores') is False,'test/online data used in confirmation')
    for field,name in [('split_sha256','split'),('training_source_manifest_sha256','training_source'),('diagnostic_source_manifest_sha256','diagnostic_source'),('selection_protocol_sha256','selection_protocol'),('frozen_postprocess_sha256','frozen_postprocess')]:require(report.get(field)==ev[name]['sha256'],'paired source/protocol binding differs '+field)
    for kind,suffix in [('control','C'),('candidate','J')]:
        part=report.get(kind,{})
        provenance=part.get('provenance',part)
        for field,name in [('identity_sha256','identity_'+suffix),('checkpoint_sha256','checkpoint_'+suffix),('checkpoint_audit_sha256','checkpoint_audit_'+suffix)]:
            require(provenance.get(field)==ev[name]['sha256'],'paired actual arm identity differs '+kind+'/'+field)
            if field in part:require(part[field]==provenance.get(field),'conflicting paired direct/nested alias '+field)
    rows={x['name']:x for x in report.get('protocols',[])}
    require({'natural_unique_images','balanced_supported_unique'}.issubset(rows),'actual registered paired protocols missing')
    natural=rows['natural_unique_images']['p7']['candidate_vs_control']['delta_vs_baseline_pp']
    primary=rows['balanced_supported_unique']['p7']['candidate_vs_control']['delta_vs_baseline_pp']
    require(math.isfinite(primary) and math.isfinite(natural),'nonfinite paired score')
    require(report.get('eligibility',{}).get('primary_delta_pp')==primary and report['eligibility'].get('natural_p7_delta_pp')==natural,'eligibility summary differs actual protocol results')
    require(primary>0,'fresh registered primary positive confirmation gate not met')
    return dict(status='pass',primary_delta_pp=primary,natural_p7_delta_pp=natural,
                primary_metric='balanced_supported_unique P7 observed-label agreement',stable_gain_claim=False)
def validate_audit(audit,identity,ev,suffix,arm):
    tensors,numel=(152,87849216) if suffix=='C' else (154,88233966)
    require(audit.get('schema')=='aictrack1.confirmation.fixed_last_diagnostic_audit.v1' and audit.get('complete') is True and audit.get('status')=='complete_fixed_last20_audit','actual complete fresh checkpoint audit missing')
    require(audit.get('model_arm')==arm and audit.get('checkpoint_role')=='last' and audit.get('checkpoint_epoch')==20 and audit.get('epoch')==20,'fixed last20 wrong arm/epoch')
    require(audit.get('trainable_tensors')==tensors and audit.get('trainable_numel')==numel and audit.get('numel',numel)==numel,'fresh parameter shape/count mismatch')
    for field,name in [('checkpoint_sha256','checkpoint_'+suffix),('fixed_last_sha256','checkpoint_'+suffix),('config_sha256','config_'+suffix),('split_sha256','split'),('h1_sha256','h1'),('training_source_manifest_sha256','training_source'),('diagnostic_source_manifest_sha256','diagnostic_source')]:require(audit.get(field)==ev[name]['sha256'],'actual audit binding differs '+field)
    require(audit.get('classifier_head_source')==('fresh_fold_h1_frozen' if suffix=='C' else 'joint_fixed_last_state') and audit.get('classifier_parameters_restored_equal') is True and audit.get('test_access') is False,'actual head restoration/source proof missing')
    require(identity.get('holdout_verified') is True and identity.get('training_validation_overlap_rows')==0 and identity.get('checkpoint_role')=='last' and identity.get('test_access') is False,'heldout frozen sidecar missing')
    for field,name in [('checkpoint_sha256','checkpoint_'+suffix),('experiment_config_sha256','config_'+suffix),('split_sha256','split'),('training_source_manifest_sha256','training_source'),('diagnostic_source_manifest_sha256','diagnostic_source')]:require(identity.get(field)==ev[name]['sha256'],'actual sidecar binding differs '+field)
def expected_ac_binding(rows,assignment):
    require(set(assignment)=={r['sample_id'] for r in rows},'new split official ID coverage differs')
    indices=[i for i,row in enumerate(rows) if assignment[row['sample_id']]=='train']
    labels=[int(rows[i]['internal_class_index']) for i in indices]
    return indices,labels
def inspect_checkpoint(path,config_sha,suffix,indices,labels):
    import torch
    torch.set_num_threads(2)
    require(Path(path).name=='last.pt','checkpoint must be exact fixed last.pt')
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(state.get('epoch')==20 and state.get('mode')=='full_visual' and state.get('training_scope')=='fixed_split_validation' and state.get('config_sha256')==config_sha and state.get('split_sha256')==SPLIT and state.get('train_manifest_sha256')==MANIFEST and state.get('clip_weight_sha256')==CLIP,'actual fixed20 checkpoint/provenance differs')
    parameters=state.get('trainable_state',{});expected=(152,87849216) if suffix=='C' else (154,88233966)
    require(len(parameters)==expected[0] and sum(x.numel() for x in parameters.values())==expected[1],'actual tensor/numel count differs')
    require(all(isinstance(x,torch.Tensor) and torch.isfinite(x).all() for x in parameters.values()),'actual trainable state nonfinite')
    head={'classifier.linear.weight','classifier.linear.bias'}
    require(all(n.startswith('visual.') or n in head for n in parameters),'unexpected new module/parameter')
    require((set(parameters)&head)==(set() if suffix=='C' else head),'actual head omitted/partial/wrong C state')
    if suffix=='J':require(parameters['classifier.linear.weight'].shape==(750,512) and parameters['classifier.linear.bias'].shape==(750,),'joint head wrong shape')
    groups=state['optimizer']['param_groups'];require([len(x['params']) for x in groups]==([152] if suffix=='C' else [152,2]),'optimizer groups differ')
    require(state['scheduler'].get('last_epoch')==20 and state['scheduler'].get('T_max')==20,'compressed/nonfinal scheduler')
    ac=state.get('confidence_weighting',{})
    require(ac.get('sample_indices') is not None and ac['sample_indices'].tolist()==indices and ac['class_ids'].tolist()==labels and ac.get('epochs_updated')==20 and torch.isfinite(ac['weights']).all(),'actual new-train global AC ID/label binding differs')
    return dict(epoch=20,checkpoint_role='last',trainable_tensors=expected[0],trainable_numel=expected[1])
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    request=read(a.request);require(request.get('schema')=='aictrack1.p1.fresh_confirmation.request.v1','request schema differs')
    arm=request.get('selected_arm');require(arm in {'J5','J20'},'exact selected J required')
    ev=evidence_files(request);require(ev['split']['sha256']==SPLIT,'must use actual new confirmation split')
    source,training_root=verify_source(ev['training_source']);verify_source(ev['diagnostic_source'])
    require(source.get('confirmation') is True and source.get('fresh_control') is True and source.get('selected_arm')==arm,'fresh confirmation source required')
    assets=read(training_root/'inputs/p1_assets.json');base=training_root.parent;split_audit=read(base/'inputs/confirmation_assets.json')
    require(assets.get('split_seed')==20261009 and assets.get('h1_row_binding')==split_audit.get('h1_row_binding') and assets['assets']['h1']['sha256']==ev['h1']['sha256'] and assets['assets']['split']['sha256']==SPLIT,'actual new train-only initialization binding differs')
    require(split_audit.get('counts')=={'train':118958,'validation':29737} and split_audit.get('train_query_connected_group_overlap')==0 and split_audit.get('old_selection_query_confirmation_group_overlap')==0 and split_audit.get('old_selection_query_rows_overlap')==0,'actual independent connected-group holdout proof missing')
    require(sha(assets['assets']['manifest']['path'])==MANIFEST,'official manifest differs')
    with Path(assets['assets']['manifest']['path']).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    with Path(ev['split']['path']).open(encoding='utf-8-sig',newline='') as f:split_rows=list(csv.DictReader(f))
    assignment={r['sample_id']:r['split'] for r in split_rows}
    require(len(assignment)==len(split_rows)==148695 and set(assignment.values())=={'train','validation'},'actual split ID/partition coverage differs')
    indices,labels=expected_ac_binding(rows,assignment);require(len(indices)==118958 and len(rows)-len(indices)==29737,'actual AC train/query counts differ')
    import yaml
    configs={suffix:yaml.safe_load(Path(ev['config_'+suffix]['path']).read_text(encoding='utf-8'))['experiment'] for suffix in ('C','J')}
    for suffix,cfg in configs.items():
        require(cfg.get('selection_rule')=='fixed_final' and cfg.get('training_scope')=='fixed_split_validation' and cfg['peft']['mode']=='full_visual' and cfg['training']['epochs']==20 and cfg['training']['seed']==20261008 and cfg['training']['visual_learning_rate']==1.75e-6 and cfg['training']['head_learning_rate']==(0 if suffix=='C' else {'J5':5e-6,'J20':2e-5}[arm]),'fresh matched fixed20 recipe differs')
        require(sha(cfg['head_checkpoint'])==ev['h1']['sha256'] and sha(cfg['split'])==SPLIT,'actual config uses other/full H1 or other split')
    matched=[]
    for cfg in configs.values():
        value=json.loads(json.dumps(cfg));value.pop('run_id');value.pop('output_dir');value['training'].pop('head_learning_rate');matched.append(value)
    require(matched[0]==matched[1],'C/J differ beyond head LR')
    runtime=validate_runtime(read(ev['runtime_C']['path']),read(ev['runtime_J']['path']),ev,arm)
    gate=validate_report(read(ev['paired_report']['path']),ev,arm)
    arms={}
    for suffix,label in [('C','C'),('J',arm)]:
        validate_audit(read(ev['checkpoint_audit_'+suffix]['path']),read(ev['identity_'+suffix]['path']),ev,suffix,label)
        detail=inspect_checkpoint(ev['checkpoint_'+suffix]['path'],ev['config_'+suffix]['sha256'],suffix,indices,labels)
        arms[label]=detail|{field:ev[name+'_'+suffix]['sha256'] for field,name in [('checkpoint_sha256','checkpoint'),('config_sha256','config'),('identity_sha256','identity'),('checkpoint_audit_sha256','checkpoint_audit')]}
    import torch
    head=torch.load(ev['h1']['path'],map_location='cpu',weights_only=True,mmap=True);h1receipt=read(ev['h1_receipt']['path'])
    require(head.get('training_scope')=='fold_train_fixed_final_epochs' and head.get('epoch')==100 and head.get('row_binding')==assets['h1_row_binding'] and head.get('best_model') is None and head['feature_cache_metadata']['training_record_count']==118958 and head['feature_cache_metadata']['split_sha256']==SPLIT,'actual H1 full/old/best/row mismatch')
    require(h1receipt.get('complete') is True and h1receipt.get('checkpoint_sha256')==ev['h1']['sha256'] and h1receipt.get('row_binding')==assets['h1_row_binding'] and h1receipt.get('validation_selection') is False and h1receipt.get('test_access') is False and h1receipt.get('actual_cpu_optimizer_scheduler_scaler_model_rng_restored') is True,'actual H1 train-only fixed100 receipt missing')
    require(sha(base/'source_manifest.json')==assets.get('source_h1_manifest_sha256')==h1receipt.get('source_sha256')==head.get('source_sha256'),'actual fixed100 H1 source chain differs')
    review=read(ev['root_review']['path']);require(review.get('status')=='pass' and review.get('reviewer')=='root' and isinstance(review.get('reason'),str) and review['reason'].strip(),'explicit independent root review required')
    payload=dict(schema='aictrack1.p1.fresh_confirmation.v1',status='positive_confirmed_for_full_training',selected_arm=arm,split_seed=20261009,split_sha256=SPLIT,
        h1_sha256=ev['h1']['sha256'],h1_receipt_sha256=ev['h1_receipt']['sha256'],confirmation_training_source_manifest_sha256=ev['training_source']['sha256'],diagnostic_source_manifest_sha256=ev['diagnostic_source']['sha256'],
        selection_protocol_sha256=ev['selection_protocol']['sha256'],frozen_postprocess_sha256=ev['frozen_postprocess']['sha256'],train_count=118958,validation_count=29737,
        initialization=dict(clip_sha256=CLIP,h1_epoch=100,h1_scope='fold_train_fixed_final_epochs'),runtime=runtime,
        train_query_overlap=0,train_query_exact_duplicate_overlap=0,train_query_near_duplicate_overlap=0,old_query_overlap=0,
        old_query_overlap_meaning='fresh query versus old selection query rows; old query may enter fresh train',used_test_or_online_scores=False,test_access=False,
        decision_gate=gate,root_review=review,evidence=ev,arms=arms,receipt_writer_sha256=sha(__file__))
    raw=(json.dumps(payload,indent=2,sort_keys=True)+'\n').encode()
    if a.output.exists():require(a.output.read_bytes()==raw,'immutable confirmation receipt already differs')
    else:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(raw)
    print(json.dumps(dict(status=payload['status'],path=str(a.output),sha256=sha(a.output),selected_arm=arm,decision_gate=gate)))
if __name__=='__main__':main()

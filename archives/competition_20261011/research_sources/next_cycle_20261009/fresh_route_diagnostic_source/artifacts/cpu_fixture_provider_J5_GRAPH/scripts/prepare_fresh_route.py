"""Create a NEW immutable bound C+one bundle from reviewed template and actual admission."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import yaml
from fresh_route_contract import ROOT,ROUTE_ARM,sha,verify_admission_evidence,validate_admission,validate_assets,expected_config,H1_SHA,H1_RECEIPT_SHA,H1_SOURCE_SHA

def verify_template(root=ROOT):
    root=Path(root);m=json.loads((root/'template_manifest.json').read_text())
    if m.get('schema')!='aictrack1.fresh_route.training.template.v1':raise ValueError('reviewed immutable template required')
    for rel,digest in m['files'].items():
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts or not (root/path).resolve().is_relative_to(root.resolve()) or sha(root/path)!=digest:raise ValueError('template member drift '+rel)
    return m

def fresh_assets(route,admission_sha,root=ROOT):
    root=Path(root);h=json.loads((root/'inputs/h1_actual_CPU_complete.json').read_text());launch=json.loads((root/'inputs/h1_actual_launch.json').read_text())
    if not (h.get('complete') is True and h.get('checkpoint_sha256')==H1_SHA and h.get('producer_receipt_sha256')==H1_RECEIPT_SHA and h.get('source_sha256')==H1_SOURCE_SHA and h.get('actual_torch_threads')==4):raise ValueError('actual new H1100 CPU acceptance required')
    old=json.loads((root/'inputs/confirmation_assets.json').read_text());deployment=json.loads((root/'inputs/h1_deployment.json').read_text())
    a=deployment['assets']|{'h1':{'path':'/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009/artifacts/experiments/confirmation_H1/last.pt','sha256':H1_SHA},
        'h1_receipt':{'path':'/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009/artifacts/experiments/confirmation_H1/h1_receipt.json','sha256':H1_RECEIPT_SHA},
        'h1_source':{'path':'/root/autodl-tmp/AICtrack1_next_cycle_confirm_20261009/source_manifest.json','sha256':H1_SOURCE_SHA}}
    payload=dict(schema='aictrack1.fresh_route.assets.v1',status='verified_new_fold_assets',competition_stage='stage2',train_count=118958,validation_count=29737,num_classes=750,training_seed=20261008,split_seed=20261009,
        dataset_root=deployment['dataset_root'],canonical_lock=deployment['canonical_lock'],assets=a,h1_training_scope='fold_train_fixed_final_epochs',h1_training_count=118958,h1_epoch=100,h1_validation_used_for_selection=False,
        h1_row_binding=h['producer_receipt']['row_binding'],terminal_cache_parity_top1=h['producer_receipt']['terminal_cache_parity_top1'],
        source_h1_manifest_sha256=H1_SOURCE_SHA,selected_route=route,selected_arm=ROUTE_ARM[route],admission_receipt_sha256=admission_sha,
        old_query_overlap=old['old_selection_query_rows_overlap'],train_query_overlap=0,
        runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','actual_torch_threads':4},test_access=False,
        original_phase_start_unix=launch['phase_budget_started_unix'],original_deadline_unix=launch['deadline_unix'],original_data_free_baseline_bytes=launch['original_data_free_baseline_bytes'])
    if old['old_selection_query_confirmation_group_overlap']!=0 or h['producer_receipt']['row_binding']!=old['h1_row_binding']:raise ValueError('real independent fresh query/H1 row binding required')
    return validate_assets(payload)

def prepare(target,admission_path,admission_sha):
    template=verify_template();target=Path(target)
    if target.exists():raise FileExistsError('bound target must be new; no mutation of reviewed template/source')
    if sha(admission_path)!=admission_sha:raise ValueError('actual admission SHA differs')
    admission=verify_admission_evidence(json.loads(Path(admission_path).read_text(encoding='utf-8')))
    payload=fresh_assets(admission['selected_route'],admission_sha)
    for rel in template['files']:
        if rel=='template_manifest.json':continue
        dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    (target/'inputs/admission_receipt.json').write_bytes(Path(admission_path).read_bytes())
    (target/'inputs/p1_assets.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    base=json.loads((target/'inputs/base_C_config.json').read_text())
    for arm in ('C',payload['selected_arm']):
        path=target/f'configs/experiment/p1_{arm}.yaml';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump({'experiment':expected_config(arm,payload,base)},sort_keys=False),encoding='utf-8')
    model=yaml.safe_load((target/'configs/model/p1_clip.yaml').read_text());model['model']['weight_path']=payload['assets']['clip']['path']
    (target/'configs/model/p1_clip.yaml').write_text(yaml.safe_dump(model,sort_keys=False),encoding='utf-8')
    data=yaml.safe_load((target/'configs/data/p1_336.yaml').read_text());data['dataset']['split']['seed']=20261009
    (target/'configs/data/p1_336.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    files={}
    for rel in template['files']:
        if (target/rel).is_file():files[rel]=sha(target/rel)
    for rel in ['inputs/admission_receipt.json','inputs/p1_assets.json']+[f'configs/experiment/p1_{x}.yaml' for x in ('C',payload['selected_arm'])]:files[rel]=sha(target/rel)
    m=dict(schema='aictrack1.fresh_route.training.source.v1',status='bound',files=files,template_manifest_sha256=sha(ROOT/'template_manifest.json'),
        selected_route=payload['selected_route'],selected_arm=payload['selected_arm'],admission_receipt_sha256=admission_sha,source_h1_manifest_sha256=H1_SOURCE_SHA,
        split_sha256=payload['assets']['split']['sha256'],h1_sha256=H1_SHA,runtime=payload['runtime'],fresh_control=True,test_access=False,
        original_phase_start_unix=1791497972.440027,original_deadline_unix=1791526772.440027,original_data_free_baseline_bytes=13013950464,disk_max_bytes=8589934592,min_free_bytes=3221225472)
    (target/'source_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return dict(status='bound_C_plus_one_typed_route_not_launched',root=str(target),selected_route=payload['selected_route'],source_manifest_sha256=sha(target/'source_manifest.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--admission-sha256',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.target,a.admission,a.admission_sha256)))

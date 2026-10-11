"""NEW bound bundle only after real fresh positive route evidence and explicit full budget."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import yaml
from full_route_contract import ROOT,sha,read,require,verify_confirmation,validate_budget,validate_assets,expected_config,FULL_H1_SHA,FULL_H1_RECEIPT_SHA

def verify_template(root=ROOT):
    root=Path(root);m=read(root/'template_manifest.json');require(m.get('schema')=='aictrack1.fresh_route.full.training.template.v1','full immutable template required')
    for rel,digest in m['files'].items():
        p=Path(rel);require(not p.is_absolute() and '..' not in p.parts and (root/p).resolve().is_relative_to(root.resolve()) and sha(root/p)==digest,'full template path/drift '+rel)
    return m

def full_assets(choice,confirmation_sha,budget_sha):
    p=read(ROOT/'inputs/legal_full_asset_reference.json')
    actual=read(ROOT/'inputs/legal_full_assets_actual_CPU_preflight.json');head=read(ROOT/'inputs/legal_full_h1_cpu_acceptance.json')
    require(actual.get('complete') is True and actual.get('actual_asset_SHA_verified') is True and actual.get('source_manifest_sha256')=='677578ba379652d8b0277f8fe3055a2dba4dbe914f76f321ff6161f2c5838262','actual677 full asset CPU acceptance required')
    require(head.get('status')=='passed' and head.get('checkpoint_sha256')==FULL_H1_SHA and head.get('epoch')==100 and head.get('training_scope')=='full_train_fixed_epochs','actual legal full H1 CPU acceptance required')
    p.update(schema='aictrack1.fresh_route.full.assets.v1',status='pinned_legal_full_assets',selected_route=choice['selected_route'],selected_arm=choice['selected_arm'],
        fresh_confirmation_receipt_sha256=confirmation_sha,full_budget_sha256=budget_sha,h1_row_binding=head['row_binding'])
    return validate_assets(p)

def prepare(target,confirmation_path,confirmation_sha,budget_path,budget_sha):
    template=verify_template();target=Path(target)
    if target.exists():raise FileExistsError('full target must be NEW; never mutate reviewed template or bound source')
    require(sha(confirmation_path)==confirmation_sha and sha(budget_path)==budget_sha,'actual full bound input SHA mismatch')
    choice=verify_confirmation(read(confirmation_path));budget=validate_budget(read(budget_path));p=full_assets(choice,confirmation_sha,budget_sha)
    for rel in template['files']:
        dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    for name,path in [('fresh_confirmation.json',confirmation_path),('full_budget.json',budget_path)]:shutil.copyfile(path,target/'inputs'/name)
    (target/'inputs/full_assets.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    arm=p['selected_arm'];config=target/f'configs/experiment/p1_full_{arm}.yaml';config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(yaml.safe_dump({'experiment':expected_config(arm,p)},sort_keys=False),encoding='utf-8')
    model=yaml.safe_load((target/'configs/model/p1_clip.yaml').read_text(encoding='utf-8'));model['model']['weight_path']=p['assets']['clip']['path']
    (target/'configs/model/p1_clip.yaml').write_text(yaml.safe_dump(model,sort_keys=False),encoding='utf-8')
    files={rel:sha(target/rel) for rel in template['files']}
    for rel in ['inputs/fresh_confirmation.json','inputs/full_budget.json','inputs/full_assets.json',f'configs/experiment/p1_full_{arm}.yaml']:files[rel]=sha(target/rel)
    m=dict(schema='aictrack1.fresh_route.full.training.source.v1',status='bound',files=files,template_manifest_sha256=sha(ROOT/'template_manifest.json'),
        selected_route=p['selected_route'],selected_arm=arm,fresh_confirmation_receipt_sha256=confirmation_sha,full_budget_sha256=budget_sha,
        full_h1_sha256=FULL_H1_SHA,full_h1_receipt_sha256=FULL_H1_RECEIPT_SHA,train_count=148695,validation_count=0,
        fixed_epoch=20,selection_rule='fixed_final',runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'},
        original_data_free_baseline_bytes=budget['initial_free_bytes'],phase_start_unix=budget['phase_start_unix'],deadline_unix=budget['deadline_unix'],disk_max_bytes=budget['disk_max_bytes'],min_free_bytes=budget['min_free_bytes'],test_access=False)
    (target/'source_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return dict(status='bound_single_full_recipe_not_launched',root=str(target),selected_route=p['selected_route'],source_manifest_sha256=sha(target/'source_manifest.json'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',type=Path,required=True);parser.add_argument('--confirmation',type=Path,required=True);parser.add_argument('--confirmation-sha256',required=True);parser.add_argument('--budget',type=Path,required=True);parser.add_argument('--budget-sha256',required=True);a=parser.parse_args()
    print(json.dumps(prepare(a.target,a.confirmation,a.confirmation_sha256,a.budget,a.budget_sha256)))

"""Bind a NEW template only to actual completed typed FULL20 files; no model/image load."""
from __future__ import annotations
import argparse,hashlib,importlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FULL_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_full_20261009')
FINAL_ROOT=Path('/root/autodl-tmp/AICtrack1_next_cycle_fresh_route_final_20261009')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,why):
    if not ok:raise ValueError(why)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def verify_manifest(folder,name,digest,schema):
    folder=Path(folder).resolve();p=folder/name;require(sha(p)==digest,'actual manifest SHA differs')
    m=read(p);require(m['schema']==schema,'manifest schema differs')
    for rel,h in m['files'].items():
        q=folder/rel;require(not Path(rel).is_absolute() and '..' not in Path(rel).parts and not q.is_symlink() and q.resolve().is_relative_to(folder) and sha(q)==h,'immutable source drift '+rel)
    return m
def prepare(args):
    template=verify_manifest(ROOT,'template_manifest.json',args.template_sha256,'aictrack1.fresh_route.final_inference.template.v1')
    require(sys.platform=='linux' and args.full_root.resolve()==FULL_ROOT.resolve(),'actual dedicated Linux FULL root required')
    require(args.target.resolve()==FINAL_ROOT.resolve() and not args.target.exists(),'one NEW dedicated final source root required')
    full=verify_manifest(args.full_root,'source_manifest.json',args.full_source_sha256,'aictrack1.fresh_route.full.training.source.v1')
    require(full.get('status')=='bound','actual bound FULL source required')
    require(full.get('template_manifest_sha256')==sha(ROOT/'references/UPSTREAM_FULL_TEMPLATE.json'),'actual FULL producer template was not reviewed')
    # Actual producer contract verifies the typed fresh-positive receipt and every evidence file.
    sys.path[:0]=[str(args.full_root/'scripts'),str(args.full_root/'src')]
    contract=importlib.import_module('p1_contract')
    payload=contract.verify_remote_assets(args.full_root)
    choice=contract.selection(args.confirmation,args.confirmation_sha256);contract.verify_selection_evidence(choice)
    route=choice['selected_route'];arm=choice['selected_arm'];require((route,arm) in [('S','S'),('J5_GRAPH','J5')],'typed selected final route only')
    require(payload.get('fresh_confirmation_receipt_sha256')==args.confirmation_sha256 and payload.get('selected_route')==route and payload.get('selected_arm')==arm,'actual FULL source uses another confirmation/route')
    require(sha(args.launch_receipt)==args.launch_receipt_sha256,'actual FULL launcher SHA differs');launch=read(args.launch_receipt)
    require(launch.get('schema')=='aictrack1.fresh_route.full.launch_result.v1' and launch.get('return_code')==0 and launch.get('synthetic_smoke') is False and launch.get('arm')==arm and launch.get('selected_route')==route and launch.get('source_manifest_sha256')==args.full_source_sha256 and launch.get('fresh_confirmation_receipt_sha256')==args.confirmation_sha256,'actual successful typed FULL run required')
    import yaml
    config=args.full_root/f'configs/experiment/p1_full_{arm}.yaml';raw=yaml.safe_load(config.read_text())['experiment'];contract.validate_config(raw,arm,payload)
    out=args.full_root/f'artifacts/experiments/FULL_{arm}';require(out.resolve()==(config.parent/raw['output_dir']).resolve(),'actual producer output path mismatch')
    require(not (out/'p1_screen_receipt.json').exists(),'partial screen receipt cannot become final')
    paths={'last':out/'last.pt','final':out/'final.pt','metrics':out/'metrics.json','config':config,'full_source':args.full_root/'source_manifest.json','selection':args.confirmation,'launch_receipt':args.launch_receipt,'full_budget':args.full_root/'inputs/full_budget.json'}
    items={k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in paths.items()};metrics=read(paths['metrics'])
    require(metrics.get('training_scope')=='full_train_fixed_epochs' and metrics.get('train_count')==148695 and metrics.get('validation_count')==0 and metrics.get('early_stop') is None,'completed official FULL148695 without validation required')
    require(metrics.get('selected_checkpoint')=='final.pt' and metrics.get('selection_rule')=='fixed final epoch 20; no validation selection' and [r.get('epoch') for r in metrics.get('history',[])]==list(range(1,21)),'fixed full20 terminal metrics required')
    require(metrics.get('selected_checkpoint_sha256')==metrics.get('final_checkpoint_sha256')==items['final']['sha256'] and metrics.get('last_checkpoint_sha256')==items['last']['sha256'] and metrics.get('config_sha256')==items['config']['sha256'],'actual terminal file hashes differ')
    require(launch.get('config_sha256')==items['config']['sha256'] and launch.get('full_budget_sha256')==items['full_budget']['sha256'],'actual launch config/budget binding differs')
    for rel in template['files']:
        q=args.target/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,q)
    binding=dict(schema='aictrack1.fresh_route.final.actual_binding.v1',selected_route=route,selected_arm=arm,full_root=str(args.full_root.resolve()),full_training_source_sha256=args.full_source_sha256,confirmation_sha256=args.confirmation_sha256,final_sha256=items['final']['sha256'],last_sha256=items['last']['sha256'],evidence=items,test_images_accessed=False,cpu_model_audit_pending=True)
    (args.target/'inputs/actual_full_binding.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8')
    files={r:sha(args.target/r) for r in template['files']};files['inputs/actual_full_binding.json']=sha(args.target/'inputs/actual_full_binding.json')
    m=dict(schema='aictrack1.fresh_route.final_inference.source.v1',status='bound_actual_full20_checkpoint',files=files,template_manifest_sha256=args.template_sha256,selected_route=route,selected_arm=arm,full_training_source_sha256=args.full_source_sha256,confirmation_sha256=args.confirmation_sha256,final_sha256=binding['final_sha256'],last_sha256=binding['last_sha256'],single_checkpoint=True,test_access=False,actual_model_cpu_audit_pending=True)
    (args.target/'source_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return dict(status='bound_actual_full20_CPU_recovery_still_required',root=str(args.target),source_manifest_sha256=sha(args.target/'source_manifest.json'),selected_route=route)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--template-sha256',required=True);p.add_argument('--full-root',type=Path,default=FULL_ROOT);p.add_argument('--full-source-sha256',required=True);p.add_argument('--confirmation',type=Path,required=True);p.add_argument('--confirmation-sha256',required=True);p.add_argument('--launch-receipt',type=Path,required=True);p.add_argument('--launch-receipt-sha256',required=True);p.add_argument('--target',type=Path,default=FINAL_ROOT);print(json.dumps(prepare(p.parse_args())));return 0
if __name__=='__main__':raise SystemExit(main())

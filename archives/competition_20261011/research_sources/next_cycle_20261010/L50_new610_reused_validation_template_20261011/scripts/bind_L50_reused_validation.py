"""Metadata binding only after real first-query positive; reuses exact completed legal H1."""
from pathlib import Path
import argparse,json,shutil,hashlib,sys,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from p1_contract import sha,read_json,verify_files,REMOTE_ROOT,expected_config,admission
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--admission',type=Path,required=True);p.add_argument('--budget',type=Path,required=True);a=p.parse_args()
 m=verify_files(require_bound=False);assert m['status']=='unbound_template' and a.output_root.resolve()==REMOTE_ROOT.resolve() and not a.output_root.exists()
 choice=admission(read_json(a.admission));b=read_json(a.budget)
 assert b['schema']=='aictrack1.next_cycle10.L50_reused_validation.budget.v1' and b['root_authorized'] is True and b['phase_start_unix']<b['deadline_unix']
 assert (b['original_initial_free_bytes'],b['capacity_added_bytes'],b['adjusted_initial_free_bytes'],b['actual_disk_total_bytes'],b['disk_max_bytes'])==(13013950464,32212254720,45226205184,128849018880,42949672960)
 assert b['min_free_bytes']>=3221225472 and b['atomic_reserve_bytes']>=2147483648
 old=Path('/root/autodl-tmp/AICtrack1_next_cycle_confirmation_visual_L35_r2_20261010')
 assert sha(old/'source_manifest.json')=='8c36fc02ecb58cd0290d13e38c43deeccef44e510cdf46f7c126bfc5fe68faf4'
 oldm=read_json(old/'source_manifest.json')
 for rel,h in oldm['files'].items():assert sha(old/rel)==h
 assets=read_json(ROOT/'inputs/actual_L35_new610_control_assets.json')
 assert assets==read_json(old/'inputs/p1_assets.json') and assets['assets']['h1']['sha256']=='4073c5b58f5993373f1678f714342d829fdb6c2d092cf60cf1ebfa7aa0083980'
 for ref in assets['assets'].values():assert sha(ref['path'])==ref['sha256']
 assets.update(selected_arm='L50',selected_route='L50_GRAPH',arms=['L50'],budget_sha256=sha(a.budget),admission_sha256=sha(a.admission))
 shutil.copytree(ROOT,a.output_root)
 def write(path,obj):path.write_text(json.dumps(obj,indent=2)+chr(10))
 shutil.copyfile(a.budget,a.output_root/'inputs/budget.json');shutil.copyfile(a.admission,a.output_root/'inputs/root_admission.json');write(a.output_root/'inputs/p1_assets.json',assets)
 cfg=a.output_root/'configs/experiment/p1_L50.yaml';cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(yaml.safe_dump({'experiment':expected_config('L50',assets,root=a.output_root)},sort_keys=False))
 m.update(status='bound',arms=['L50'],selected_candidate='L50',actual_budget_sha256=sha(a.budget),actual_admission_sha256=sha(a.admission),source_h1_manifest_sha256=assets['source_h1_manifest_sha256'],h1_checkpoint_sha256=assets['assets']['h1']['sha256'],training_seed=20261010,training_scope='fixed_split_validation',fixed_epochs=20,checkpoint_role='last',future_checkpoint_sha256={'L50':None},template_manifest_sha256=sha(ROOT/'source_manifest.json'))
 m['files']={f.relative_to(a.output_root).as_posix():sha(f) for f in sorted(a.output_root.rglob('*')) if f.is_file() and f.name!='source_manifest.json'};write(a.output_root/'source_manifest.json',m)
 print(json.dumps(dict(complete=True,scope='metadata binding only',source_sha256=sha(a.output_root/'source_manifest.json'),future_last_sha256=None,GPU_started=False)))
if __name__=='__main__':main()

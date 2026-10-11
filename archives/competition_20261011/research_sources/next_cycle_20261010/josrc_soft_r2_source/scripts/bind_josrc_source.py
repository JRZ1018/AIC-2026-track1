"""Create a new exact B profile or B production root with an actual budget; no GPU."""
import argparse,json,os,shutil,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p1_contract import PROFILE_ROOT,TRAIN_ROOT,sha,assets,expected_config,require

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','train'],required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--budget',type=Path,required=True);a=p.parse_args();m=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'));require(m.get('schema')=='aictrack1.josrc_inspired.source.v1' and m['status']=='unbound_template','frozen unbound B source required')
 for rel,d in m['files'].items():require(sha(ROOT/rel)==d,'template drift '+rel)
 expected=PROFILE_ROOT if a.mode=='profile' else TRAIN_ROOT;require(os.name=='posix' and a.output_root.resolve()==expected.resolve() and not a.output_root.exists(),'new dedicated B mode root required')
 b=json.loads(a.budget.read_text(encoding='utf-8'));require(b.get('schema') in ['aictrack1.supervision_joint.budget.v1','aictrack1.josrc_inspired.budget.v1'] and b.get('root_authorized') is True and b.get('test_access') is False and b.get('final_gpu_reserve_hours')==4,'actual approved profile/train clock with final reserve')
 for key,v in dict(initial_free_bytes=13013950464,capacity_increase_bytes=32212254720,adjusted_initial_free_bytes=45226205184,data_total_bytes=128849018880,min_free_bytes=3221225472,atomic_reserve_bytes=2147483648).items():require(b.get(key)==v,'no resource ledger reset '+key)
 require(isinstance(b.get('disk_max_bytes'),int) and 21474836480<=b['disk_max_bytes']<=32212254720,'root explicit cumulative20..30GiB cap required')
 require(b['phase_start_unix']<b['deadline_unix'],'actual stage clock required');payload=assets(ROOT)
 for key,item in payload['assets'].items():require(sha(item['path'])==item['sha256'],'actual official input drift '+key)
 a.output_root.mkdir(parents=True,exist_ok=False)
 for rel in m['files']:
  target=a.output_root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
 shutil.copyfile(ROOT/'source_manifest.json',a.output_root/'template_manifest.json');shutil.copyfile(a.budget,a.output_root/'inputs/budget.json')
 payload.update(execution_root=str(expected),profile_only=a.mode=='profile',actual_budget_sha256=sha(a.budget));(a.output_root/'inputs/p1_assets.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 for arm in ('B25','B50'):
  target=a.output_root/f'configs/experiment/p1_{arm}.yaml';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(yaml.safe_dump({'experiment':expected_config(arm,payload,root=a.output_root)},sort_keys=False),encoding='utf-8')
 bound=dict(schema='aictrack1.josrc_inspired.source.v1',status='bound',mode=a.mode,profile_only=a.mode=='profile',arms=['B25','B50'],first_coefficient=.25,training_seed=20261008,heldout_H1_sha256=payload['assets']['h1']['sha256'],template_manifest_sha256=sha(ROOT/'source_manifest.json'),actual_budget_sha256=sha(a.budget),future_checkpoint_sha256={'B25':None,'B50':None},GPU_started=False,test_access=False,files={p.relative_to(a.output_root).as_posix():sha(p) for p in sorted(a.output_root.rglob('*')) if p.is_file()})
 (a.output_root/'source_manifest.json').write_text(json.dumps(bound,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(dict(bound=True,mode=a.mode,source_manifest_sha256=sha(a.output_root/'source_manifest.json'),GPU_started=False)));return 0
if __name__=='__main__':raise SystemExit(main())

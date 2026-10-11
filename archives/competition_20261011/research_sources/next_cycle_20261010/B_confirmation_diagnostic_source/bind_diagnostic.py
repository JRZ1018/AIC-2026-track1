"""CPU metadata binding to actual B producer/budget/launch; never reads a writing checkpoint."""
import argparse,json,os,shutil
from pathlib import Path
from fresh_diag_contract import ROOT,TRAIN_ROOT,REMOTE_ROOT,sha,valid_digest,verify_training_binding,save_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--training-root',type=Path,required=True);p.add_argument('--expected-training-source-sha256',required=True);p.add_argument('--actual-training-launch',type=Path,required=True);p.add_argument('--expected-launch-sha256',required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
 if manifest.get('status')!='unbound_template':raise ValueError('exclusive frozen diagnostic template only')
 for rel,digest in manifest['files'].items():
  path=(ROOT/rel).resolve()
  if '..' in Path(rel).parts or not path.is_relative_to(ROOT) or sha(path)!=digest:raise ValueError('template drift/escape '+rel)
 if os.name!='posix' or a.training_root.resolve()!=TRAIN_ROOT.resolve() or a.output_root.resolve()!=REMOTE_ROOT.resolve() or a.output_root.exists():raise ValueError('actual dedicated producer and new diagnostic root')
 assets,source=verify_training_binding(a.training_root,a.expected_training_source_sha256,runtime_assets=True,binding_only=True)
 if sha(a.actual_training_launch)!=a.expected_launch_sha256:raise ValueError('actual launch SHA required')
 launch=json.loads(a.actual_training_launch.read_text(encoding='utf-8'))
 if launch.get('mode','train')!='train' or launch.get('bound_source_sha256')!=a.expected_training_source_sha256 or launch.get('test_access') is not False:raise ValueError('actual B production launch, not profile')
 budget=a.training_root/'inputs/budget.json';b=json.loads(budget.read_text(encoding='utf-8'))
 if source['actual_budget_sha256']!=sha(budget) or assets['budget_sha256']!=sha(budget) or b.get('schema')!='aictrack1.next_cycle10.confirmation_budget.v1' or b.get('root_authorized') is not True :raise ValueError('same actual B train+validation budget source')
 from phase_guard import actual_command_arms,normalize_owner
 actual_command_arms(launch.get('command'),a.training_root,assets['selected_arm']);normalize_owner(launch['process'])
 a.output_root.mkdir(parents=True,exist_ok=False)
 for rel in manifest['files']:
  target=a.output_root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
 shutil.copyfile(ROOT/'source_manifest.json',a.output_root/'template_manifest.json');shutil.copyfile(a.training_root/'source_manifest.json',a.output_root/'inputs/training_source_manifest.json');shutil.copyfile(budget,a.output_root/'inputs/budget.json');shutil.copyfile(a.actual_training_launch,a.output_root/'inputs/actual_training_launch_snapshot.json')
 binding=json.loads((ROOT/'inputs/diagnostic_binding.json').read_text(encoding='utf-8'));binding.update(status='bound',training_source_sha256=a.expected_training_source_sha256,budget_sha256=sha(budget),actual_launch_sha256=a.expected_launch_sha256,h1_checkpoint_sha256=assets['assets']['h1']['sha256'],h1_source_sha256=assets['source_h1_manifest_sha256'],selected_candidate=assets['selected_arm'])
 save_json(a.output_root/'inputs/diagnostic_binding.json',binding)
 save_json(a.output_root/'inputs/actual_training_launch_reference.json',dict(remote_path=str(a.actual_training_launch.resolve()),remote_sha256=a.expected_launch_sha256,local_snapshot='actual_training_launch_snapshot.json',local_snapshot_sha256=sha(a.actual_training_launch)))
 bound=dict(manifest,status='bound',template_manifest_sha256=sha(ROOT/'source_manifest.json'),binding=binding,files={p.relative_to(a.output_root).as_posix():sha(p) for p in sorted(a.output_root.rglob('*')) if p.is_file()})
 save_json(a.output_root/'source_manifest.json',bound);print(json.dumps(dict(bound=True,diagnostic_source_sha256=sha(a.output_root/'source_manifest.json'),actual_training_source_sha256=a.expected_training_source_sha256,checkpoint_read=False,GPU_started=False,test_access=False)));return 0
if __name__=='__main__':raise SystemExit(main())

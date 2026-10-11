"""Bind actual AFA metadata only; active last/model/image is never opened."""
import argparse,json,os,shutil
from pathlib import Path
from fresh_diag_contract import ROOT,TRAIN_ROOT,REMOTE_ROOT,sha,verify_training_binding,save_json,verify_owned_sources
from phase_guard import validate_budget_payload,validate_actual_launch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True);p.add_argument('--expected-training-source-sha256',required=True);p.add_argument('--actual-training-launch',type=Path,required=True);p.add_argument('--expected-launch-sha256',required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
 verify_owned_sources()
 if manifest.get('status')!='unbound_template':raise ValueError('frozen unbound template only')
 if os.name!='posix' or a.training_root.resolve()!=TRAIN_ROOT.resolve() or a.output_root.resolve()!=REMOTE_ROOT.resolve() or a.output_root.exists():raise ValueError('actual AFA producer and new exclusive diagnostic root')
 assets,source=verify_training_binding(a.training_root,a.expected_training_source_sha256,runtime_assets=True,binding_only=True)
 if sha(a.actual_training_launch)!=a.expected_launch_sha256:raise ValueError('actual observed launch SHA')
 launch=json.loads(a.actual_training_launch.read_text(encoding='utf-8'));controller=validate_actual_launch(launch,a.expected_training_source_sha256,budget_sha=sha(a.training_root/'inputs/budget.json'))
 if controller.get('cwd')!=TRAIN_ROOT.as_posix():raise ValueError('actual AFA controller cwd')
 budget=a.training_root/'inputs/budget.json'
 budget_sha=sha(budget)
 if source.get('budget_sha256')!=budget_sha:raise ValueError('actual new bound training budget SHA')
 b=json.loads(budget.read_text(encoding='utf-8'))
 validate_budget_payload(b)
 if 'budget_sha256' in launch and launch['budget_sha256']!=budget_sha:raise ValueError('actual AFA launch budget SHA')
 a.output_root.mkdir(parents=True,exist_ok=False)
 for rel in manifest['files']:
  target=a.output_root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
 shutil.copyfile(ROOT/'source_manifest.json',a.output_root/'template_manifest.json');shutil.copyfile(a.training_root/'source_manifest.json',a.output_root/'inputs/training_source_manifest.json');shutil.copyfile(budget,a.output_root/'inputs/budget.json');shutil.copyfile(a.actual_training_launch,a.output_root/'inputs/actual_training_launch_snapshot.json')
 binding=json.loads((ROOT/'inputs/diagnostic_binding.json').read_text(encoding='utf-8'));binding.update(status='bound',training_source_sha256=a.expected_training_source_sha256,budget_sha256=budget_sha,actual_launch_sha256=a.expected_launch_sha256)
 save_json(a.output_root/'inputs/diagnostic_binding.json',binding)
 save_json(a.output_root/'inputs/actual_training_launch_reference.json',dict(remote_path=str(a.actual_training_launch.resolve()),remote_sha256=a.expected_launch_sha256,local_snapshot='actual_training_launch_snapshot.json',local_snapshot_sha256=sha(a.actual_training_launch)))
 bound=dict(manifest,status='bound',template_manifest_sha256=sha(ROOT/'source_manifest.json'),binding=binding,training_source_manifest_sha256=a.expected_training_source_sha256,shared_budget_sha256=budget_sha,actual_launch_sha256=a.expected_launch_sha256,files={p.relative_to(a.output_root).as_posix():sha(p) for p in sorted(a.output_root.rglob('*')) if p.is_file()})
 save_json(a.output_root/'source_manifest.json',bound)
 print(json.dumps(dict(bound=True,diagnostic_source_sha256=sha(a.output_root/'source_manifest.json'),actual_training_source_sha256=a.expected_training_source_sha256,checkpoint_read=False,model_loaded=False,GPU_started=False,images_opened=False,test_access=False)));return 0
if __name__=='__main__':raise SystemExit(main())

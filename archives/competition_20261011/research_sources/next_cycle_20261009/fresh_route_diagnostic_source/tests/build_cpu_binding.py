"""Build synthetic CPU declarations around a real provider copy, never admit a production recipe."""
from pathlib import Path
import importlib.util,hashlib,json,shutil,sys,yaml
ROOT=Path(__file__).resolve().parents[1]
PROVIDER=ROOT.parent/'fresh_route_training_source'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def build(route='S'):
 target=ROOT/('artifacts/cpu_fixture_provider_'+route);target.mkdir(parents=True,exist_ok=True)
 files=[]
 for path in PROVIDER.rglob('*'):
  if not path.is_file() or any(part in ('__pycache__','artifacts') for part in path.relative_to(PROVIDER).parts) or path.name.endswith('manifest.json'):continue
  rel=path.relative_to(PROVIDER);out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,out);files.append(rel.as_posix())
 spec=importlib.util.spec_from_file_location('cpu_prepare_fresh_route',PROVIDER/'scripts/prepare_fresh_route.py');module=importlib.util.module_from_spec(spec)
 sys.path.insert(0,str(PROVIDER/'scripts'));spec.loader.exec_module(module)
 arm=module.ROUTE_ARM[route]
 admission=dict(schema='aictrack1.fresh_route.admission.v1',status='selected_recipe_for_fresh_route_confirmation',selected_route=route,selected_arm=arm,test_access=False,used_test_or_online_scores=False,root_reviewed=True,selection_split_sha256=module.OLD_SPLIT_SHA if hasattr(module,'OLD_SPLIT_SHA') else '0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c',checkpoint_epoch=20,checkpoint_role='last',training_scope='fixed_split_validation',primary_scope='S_P7_vs_C' if route=='S' else 'J5_graph_P7_vs_C',primary_delta_vs_C_pp=.2)
 receipt=target/'inputs/admission_receipt.json';receipt.write_text(json.dumps(admission)+'\n')
 assets=module.fresh_assets(route,digest(receipt),root=PROVIDER)
 (target/'inputs/p1_assets.json').write_text(json.dumps(assets)+'\n')
 base=json.loads((target/'inputs/base_C_config.json').read_text())
 for arm in ('C',module.ROUTE_ARM[route]):
  path=target/f'configs/experiment/p1_{arm}.yaml';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump({'experiment':module.expected_config(arm,assets,base)},sort_keys=False));files.append(path.relative_to(target).as_posix())
 files+=['inputs/admission_receipt.json','inputs/p1_assets.json']
 manifest=dict(schema='aictrack1.fresh_route.training.source.v1',status='bound',selected_route=route,selected_arm=module.ROUTE_ARM[route],admission_receipt_sha256=digest(receipt),source_h1_manifest_sha256=module.H1_SOURCE_SHA,files={f:digest(target/f) for f in sorted(set(files))},original_phase_start_unix=1791497972.440027,original_deadline_unix=1791526772.440027,original_data_free_baseline_bytes=13013950464,disk_max_bytes=8589934592,min_free_bytes=3221225472,synthetic_CPU_declarations_only=True,test_access=False,template_manifest_sha256=digest(PROVIDER/'template_manifest.json'))
 (target/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 value=dict(training_root=str(target.resolve()),training_source_sha256=digest(target/'source_manifest.json'),production_admission_not_verified=True,remote_assets_not_read=True,provider_source_files={f:digest(PROVIDER/f) for f in sorted(set(files)) if (PROVIDER/f).is_file()})
 (ROOT/('artifacts/cpu_fixture_binding_'+route+'.json')).write_text(json.dumps(value,indent=2)+'\n');return value
if __name__=='__main__':
 values={route:build(route) for route in ('S','J5_GRAPH')}
 (ROOT/'artifacts/cpu_fixture_binding.json').write_text(json.dumps(values['S'],indent=2)+'\n')
 print(json.dumps({route:dict(training_root=item['training_root'],training_source_sha256=item['training_source_sha256']) for route,item in values.items()}))

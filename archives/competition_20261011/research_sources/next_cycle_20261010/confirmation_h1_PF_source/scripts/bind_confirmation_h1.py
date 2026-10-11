"""Root-only CPU binder creates a NEW execution directory; no future clocks or hashes."""
from __future__ import annotations
import argparse,json,os,shutil,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from confirmation_contract import verify_source,normalized_assets,sha,read_json,require,DEPLOYMENT_ROOT,LOCK,HEAD_CONFIG,validate_budget,validate_admission,audit_class_mapping
from cpu_feature_preflight import cpu_audit

def main():
    p=argparse.ArgumentParser()
    for name in ['output-root','manifest','class-mapping','label-conflicts','clip','feature-cache','dataset-root','budget','admission']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=verify_source(require_bound=False);require(source['status']=='unbound_template','template only')
    require(os.name=='posix' and a.output_root.resolve()==DEPLOYMENT_ROOT.resolve() and not a.output_root.exists(),'new exact dedicated Linux output ROOT required')
    require(a.dataset_root.is_absolute() and (a.dataset_root/'train').is_dir(),'actual official train root required')
    budget=validate_budget(read_json(a.budget));admission=validate_admission(read_json(a.admission));audit=normalized_assets()
    expected={'manifest':audit['assets']['manifest']['sha256'],'class_mapping':'0f54c471945f356177c885ba2f9edc943d669a61f12c73503fd4632c91ef40be','clip':'40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'}
    assets={}
    for name,path in [('manifest',a.manifest),('class_mapping',a.class_mapping),('label_conflicts',a.label_conflicts),('clip',a.clip)]:
        require(path.is_absolute() and path.is_file(),'absolute actual asset required '+name);actual=sha(path)
        if name in expected:require(actual==expected[name],'actual asset SHA differs '+name)
        assets[name]={'path':str(path.resolve()),'sha256':actual}
    mapping_identity=audit_class_mapping(a.class_mapping)
    cpu=cpu_audit(a.manifest,a.feature_cache,a.class_mapping)
    a.output_root.mkdir(parents=True,exist_ok=False)
    for rel in source['files']:
        target=a.output_root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    shutil.copyfile(ROOT/'source_manifest.json',a.output_root/'template_manifest.json')
    shutil.copyfile(a.budget,a.output_root/'inputs/budget.json');shutil.copyfile(a.admission,a.output_root/'inputs/root_admission.json')
    assets['split']={'path':str(a.output_root/'inputs/confirmation_split.csv'),'sha256':audit['assets']['split']['sha256']}
    deployment=dict(schema='aictrack1.next_cycle10.confirmation.h1.deployment.v1',assets_receipt_sha256=sha(a.output_root/'inputs/confirmation_assets.json'),canonical_lock=LOCK,class_mapping_identity=mapping_identity,dataset_root=str(a.dataset_root.resolve()),feature_cache=str(a.feature_cache.resolve()),assets=assets,budget_sha256=sha(a.output_root/'inputs/budget.json'),admission_sha256=sha(a.output_root/'inputs/root_admission.json'),runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_num_threads':4},test_access=False)
    (a.output_root/'inputs/h1_deployment.json').write_text(json.dumps(deployment,indent=2)+'\n',encoding='utf-8')
    config={'experiment':dict(run_id='20261010-confirmation-H1',execution_enabled=True,training_scope='fold_train_fixed_final_epochs',training=HEAD_CONFIG,output_dir='../../artifacts/experiments/confirmation_H1',device='cuda')}
    (a.output_root/'configs/experiment/confirmation_H1.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (a.output_root/'inputs/actual_bound_CPU_preflight.json').write_text(json.dumps(cpu,indent=2)+'\n',encoding='utf-8')
    files={p.relative_to(a.output_root).as_posix():sha(p) for p in sorted(a.output_root.rglob('*')) if p.is_file()}
    manifest=dict(schema='aictrack1.next_cycle10.PF_confirmation.h1.source.v1',status='bound_h1_only',files=files,template_manifest_sha256=sha(ROOT/'source_manifest.json'),query_definition_source_sha256=source['query_definition_source_sha256'],H1_checkpoint_sha256=None,GPU_started=False,test_access=False)
    (a.output_root/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='bound_h1_only',output_root=str(a.output_root),source_manifest_sha256=sha(a.output_root/'source_manifest.json'),CPU_preflight_complete=True,GPU_started=False)));return 0
if __name__=='__main__':raise SystemExit(main())

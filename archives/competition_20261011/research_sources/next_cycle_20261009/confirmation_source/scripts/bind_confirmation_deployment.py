"""Root runs before H1 launch; verifies actual Linux paths then freezes a new package."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import yaml
from confirmation_contract import ROOT,LOCK,HEAD_CONFIG,DEPLOYMENT_ROOT,sha,read_json,write_new
def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','class-mapping','label-conflicts','clip','feature-cache','dataset-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit=read_json(ROOT/'inputs/confirmation_assets.json')
    if os.name=='nt':raise RuntimeError('root must bind real Linux server paths; local receipt is not remote proof')
    if ROOT.resolve()!=DEPLOYMENT_ROOT:raise ValueError('exact data-disk confirmation ROOT required; no cross-disk output symlink')
    if not a.dataset_root.is_absolute() or not (a.dataset_root/'train').is_dir():raise ValueError('actual absolute Stage2 train image root required')
    assets={}
    for name,path in [('manifest',a.manifest),('class_mapping',a.class_mapping),('label_conflicts',a.label_conflicts),('clip',a.clip),('split',ROOT/'inputs/confirmation_split.csv')]:
        expected='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af' if name=='clip' else audit['assets'][name]['sha256']
        if not path.is_absolute() or sha(path)!=expected:raise ValueError('actual remote asset SHA/path differs '+name)
        assets[name]=dict(path=str(path.resolve()),sha256=expected)
    identity=read_json(ROOT/'inputs/feature_identity.json')
    if sha(a.feature_cache/'metadata.json')!=identity['metadata_sha256'] or sha(a.feature_cache/'complete.json')!=identity['complete_sha256']:raise ValueError('actual immutable official feature cache differs')
    for item in identity['shards']:
        if sha(a.feature_cache/'shards'/item['name'])!=item['sha256']:raise ValueError('remote feature shard drift')
    d=dict(schema='aictrack1.confirmation.h1.deployment.v1',assets_receipt_sha256=sha(ROOT/'inputs/confirmation_assets.json'),
        canonical_lock=LOCK,dataset_root=str(a.dataset_root.resolve()),feature_cache=str(a.feature_cache.resolve()),assets=assets,
        runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_num_threads':4},test_access=False)
    write_new(ROOT/'inputs/h1_deployment.json',d)
    (ROOT/'configs/experiment/confirmation_H1.yaml').write_text(yaml.safe_dump({'experiment':dict(run_id='20261009-confirmation-H1',execution_enabled=True,training_scope='fold_train_fixed_final_epochs',training=HEAD_CONFIG,output_dir='../../artifacts/experiments/confirmation_H1',device='cuda')},sort_keys=False),encoding='utf-8')
    files={}
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if path.is_file() and not {'artifacts','__pycache__','prepared_visual','test_fixtures','official_feature_cache','receipt_source'}.intersection(rel.parts) and path.name not in {'source_manifest.json','DEPLOYMENT_FILES.json','CONFIRMATION_DELIVERY.json'} and path.suffix in {'.py','.yaml','.json','.csv','.txt','.md'}:files[rel.as_posix()]=sha(path)
    write_new(ROOT/'source_manifest.json',dict(schema='aictrack1.confirmation.h1.source.v1',status='bound_h1_only',files=files,
        upstream_source_manifest_sha256=read_json(ROOT/'clone_receipt.json')['source_manifest_sha256'],test_access=False))
    print(json.dumps({'status':'bound_h1_only','source_sha256':sha(ROOT/'source_manifest.json')}))
if __name__=='__main__':main()

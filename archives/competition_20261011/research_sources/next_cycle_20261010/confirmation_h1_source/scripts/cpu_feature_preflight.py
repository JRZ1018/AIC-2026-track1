"""Real CPU-only full-row official-cache/new-fold audit. No image decode or fitting."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from confirmation_contract import verify_source,normalized_assets,load_rebound_cache,audit_external_cache,sha,require

def cpu_audit(manifest,feature_cache):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(4);start=time.monotonic();verify_source(require_bound=False);a=normalized_assets()
    require(sha(manifest)==a['assets']['manifest']['sha256'],'actual official manifest SHA differs')
    audit_external_cache(feature_cache)
    cache,binding=load_rebound_cache({'feature_cache':str(feature_cache),'assets':{'manifest':{'path':str(manifest)}}},a)
    train=[i for i,p in enumerate(cache.partitions) if p=='train'];query=[i for i,p in enumerate(cache.partitions) if p=='validation']
    require(len(train)==118958 and len(query)==29737 and torch.bincount(cache.labels[train],minlength=750).min()>0 and torch.bincount(cache.labels[query],minlength=750).min()>0,'actual new-fold counts/class support differ')
    return dict(schema='aictrack1.next_cycle10.h1.actual_CPU_preflight.v1',complete=True,source_manifest_sha256=sha(ROOT/'source_manifest.json'),query_definition_source_sha256=sha(ROOT/'inputs/query_definition_source_manifest.json'),row_binding=binding,feature_identity_sha256=sha(ROOT/'inputs/feature_identity.json'),actual_rows=148695,actual_shards=37,train_count=len(train),validation_count=len(query),row_fields_verified=['ID','label','index','path'],features_finite_normalized=True,H1_checkpoint_sha256=None,GPU_started=False,training_started=False,images_decoded=False,test_access=False,actual_torch_threads=torch.get_num_threads(),seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--feature-cache',type=Path,required=True);p.add_argument('--receipt',type=Path);a=p.parse_args();result=cpu_audit(a.manifest,a.feature_cache)
    if a.receipt:
        a.receipt.parent.mkdir(parents=True,exist_ok=True)
        with a.receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())

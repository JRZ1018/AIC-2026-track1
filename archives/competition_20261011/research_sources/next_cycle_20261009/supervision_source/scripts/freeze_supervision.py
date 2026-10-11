"""Freeze this independent text source; explicitly excludes runtime outputs and CPU result."""
import json
from pathlib import Path
from p1_contract import ROOT,sha,assets,validate_config
import yaml

def main():
    validate_config(yaml.safe_load((ROOT/'configs/experiment/p1_S.yaml').read_text())['experiment'],'S',assets())
    files={}
    for parent in ('scripts','src','configs','inputs','tests'):
        for p in sorted((ROOT/parent).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.yaml','.json','.txt','.md'}:
                files[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('README.md','requirements-lock.txt'):files[name]=sha(ROOT/name)
    manifest=dict(schema='aictrack1.supervision.source.v1',status='bound',files=files,
        upstream_source_manifest_sha256='2fee29fc5e5c282a331883de2b3cd4d5865c28e38a8cead89be247ec95a6c9fe',
        model_arm='S',parent_arm='C',single_algorithm_change='SCE alpha .1 to.5; all other registered C settings unchanged',
        source_only=True,checkpoint_or_data_copied=False,test_access=False,
        remote_root='/root/autodl-tmp/AICtrack1_next_cycle_supervision_20261009',
        runtime=dict(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',actual_torch_threads_required=4),
        trainable_tensors=152,trainable_numel=87849216,classifier_head_source='original_fold_H1_frozen')
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(source_manifest_sha256=sha(ROOT/'source_manifest.json'),files=len(files))))
if __name__=='__main__':main()

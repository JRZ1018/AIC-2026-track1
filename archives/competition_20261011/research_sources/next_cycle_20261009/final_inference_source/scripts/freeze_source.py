"""Freeze only this independent inference package's whitelisted code and inputs."""
import ast,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files={}
    # Do not recurse temporary ACL directories, runtime outputs or model assets.
    for relative in ('src','scripts','references','inputs'):
        for p in sorted((ROOT/relative).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
                files[p.relative_to(ROOT).as_posix()]=sha(p)
    for relative in ('tests/test_final_contract.py','requirements-lock.txt','README.md','CPU_VERIFICATION.json'):
        p=ROOT/relative;files[relative]=sha(p)
    value=dict(schema='aictrack1.p1.final_inference_source.v1',full_training_source_sha256='677578ba379652d8b0277f8fe3055a2dba4dbe914f76f321ff6161f2c5838262',
        status='engineering_ready_actual_full20_checkpoint_cpu_audit_required',files=files,
        future_checkpoint_sha256=None,checkpoint_sha_strategy='derived only after actual completed metrics/last20/final20 verification',
        test_access=False,gpu_access=False,single_checkpoint=True,
        inference=dict(native_resolution=336,extra_resize=384,view_weights=[.25]*4,postprocess='P7',temperature_multiplier=.5,bias_strength=.75,iterations=60),
        copied_full_modules_unchanged=True,optional_feature_hook_extra_forwards=0,
        runtime={'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    target=ROOT/'source_manifest.json'
    with target.open('x',encoding='utf-8',newline='\n') as handle:json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps(dict(complete=True,source_manifest_sha256=sha(target),files=len(files),bytes=sum((ROOT/r).stat().st_size for r in files))))
if __name__=='__main__':main()

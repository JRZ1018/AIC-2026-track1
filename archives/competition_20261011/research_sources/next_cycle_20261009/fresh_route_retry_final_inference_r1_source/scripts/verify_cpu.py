"""One bounded CPU engineering verification; no actual model/checkpoint/image access."""
from __future__ import annotations
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_FULL='578eb15ce02421e16e3476b063c406fdf0322d8cc63233e740c29d9f51ab4d71'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=ROOT/'CPU_VERIFICATION.json'
    if target.exists():raise FileExistsError('immutable verification receipt already exists')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    commands=[([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],'cpu_tests.log'),
        ([sys.executable,'-B','scripts/check_full_provider.py','--full-template',str(ROOT.parent/'fresh_route_full_expanded_training_source'),'--expected-template-sha256',EXPECTED_FULL],'actual_full_provider_cpu.log')]
    (ROOT/'artifacts').mkdir(exist_ok=True);results=[];started=time.time()
    for command,log in commands:
        result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True,timeout=180)
        output=result.stdout+'\n'+result.stderr
        with (ROOT/'artifacts'/log).open('x',encoding='utf-8') as f:f.write(output)
        results.append(dict(command=command,returncode=result.returncode,log_path='artifacts/'+log,log_sha256=sha(ROOT/'artifacts'/log)))
        print(json.dumps(dict(command=command,returncode=result.returncode,output=output)),flush=True)
        if result.returncode:raise RuntimeError('bounded actual CPU verification failed; log preserved')
    log=(ROOT/'artifacts/cpu_tests.log').read_text(encoding='utf-8')
    if 'Ran 24 tests' not in log or '\nOK\n' not in log:raise ValueError('24 actual CPU fixtures must complete')
    provider=json.loads((ROOT/'artifacts/actual_full_provider_cpu.log').read_text())
    if provider.get('full_template_manifest_sha256')!=EXPECTED_FULL:raise ValueError('actual provider template receipt mismatch')
    tested={p.relative_to(ROOT).as_posix():sha(p) for base in ('scripts','src','tests') for p in sorted((ROOT/base).rglob('*.py')) if '__pycache__' not in p.parts}
    receipt=dict(schema='aictrack1.fresh_route.final_capacity.cpu_verification.v1',status='PASS',started_unix=started,completed_unix=time.time(),
        test_count=24,actual_full_provider_template_sha256=EXPECTED_FULL,commands=results,tested_python_files=tested,
        actual_checkpoint_or_official_model_loaded=False,test_images_opened=False,server_access=False,gpu_access=False,
        future_full_training_source_sha256=None,future_checkpoint_sha256=None,user_explicit_test_authorization=True,
        authorization_source='direct human instruction relayed by root: 我明确授予你所有权限，包括测试授权')
    with target.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(receipt_sha256=sha(target),test_count=24,actual_full_provider_template_sha256=EXPECTED_FULL)))
    return 0
if __name__=='__main__':raise SystemExit(main())

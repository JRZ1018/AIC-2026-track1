"""Only fresh C plus the admitted single candidate; existing canonical queues own leases."""
import argparse,json,os,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fresh_route_contract import assets,verify_files,verify_remote_assets,sha
from fresh_route_runtime import RUNTIME_ENV,bind_budget

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume-arms',nargs='*',default=[])
    a=p.parse_args();verify_files();payload=verify_remote_assets();order=['C',payload['selected_arm']]
    if not set(a.resume_arms).issubset(order):p.error('resume only C and selected arm')
    bind_budget(ROOT,start=1791497972.440027,deadline=1791526772.440027,initial=13013950464,diskmax=8589934592,minimum=3221225472)
    results=[];env=os.environ.copy();env.update(RUNTIME_ENV)
    for arm in order:
        command=[sys.executable,'scripts/run_fresh_route_arm.py','--arm',arm]
        if arm in a.resume_arms:command.append('--resume')
        child=subprocess.Popen(command,cwd=ROOT,env=env)
        try:code=child.wait(timeout=max(.1,1791526772.440027-time.time())+45)
        except BaseException:
            # Arm controller owns a separate trainer session and its exact cleanup.
            # Signal the controller's interpreter so its wait catches the interruption.
            child.send_signal(__import__('signal').SIGINT)
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                raise RuntimeError('fresh arm controller did not finish cleanup; retain canonical lock and inspect exact process')
            raise
        results.append(dict(arm=arm,return_code=code))
        receipt=ROOT/'artifacts/queues'/f'fresh_route_pair_attempt_{uuid.uuid4().hex}.json';receipt.parent.mkdir(parents=True,exist_ok=True)
        with receipt.open('x',encoding='utf-8') as handle:json.dump(dict(schema='aictrack1.fresh_route.pair_attempt.v1',selected_route=payload['selected_route'],selected_arm=payload['selected_arm'],results=results,source_manifest_sha256=sha(ROOT/'source_manifest.json'),test_access=False),handle,indent=2)
        if code:return code
    return 0
if __name__=='__main__':raise SystemExit(main())

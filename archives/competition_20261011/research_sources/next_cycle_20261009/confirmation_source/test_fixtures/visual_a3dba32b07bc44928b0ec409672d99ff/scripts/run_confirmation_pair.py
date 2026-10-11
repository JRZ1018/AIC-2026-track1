"""Fresh C and one independently selected J; original serial queue owns each lease."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
from p1_contract import ROOT,ARMS,assets,verify_files
def main():
    p=argparse.ArgumentParser();p.add_argument('--resume-arms',nargs='*',choices=list(ARMS),default=[])
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--initial-free-bytes',type=int,required=True)
    p.add_argument('--disk-max-bytes',type=int,default=20*1024**3);p.add_argument('--min-free-bytes',type=int,default=3*1024**3)
    a=p.parse_args();verify_files();payload=assets()
    if set(ARMS)!={'C',payload['selected_arm']}:raise ValueError('exact fresh C and single selected J required')
    results=[]
    for arm in ['C',payload['selected_arm']]:
        command=[sys.executable,'scripts/run_p1.py','--arm',arm,'--deadline-unix',str(a.deadline_unix),'--initial-free-bytes',str(a.initial_free_bytes),'--disk-max-bytes',str(a.disk_max_bytes),'--min-free-bytes',str(a.min_free_bytes)]
        if arm in a.resume_arms:command.append('--resume')
        code=subprocess.call(command,cwd=ROOT);results.append(dict(arm=arm,return_code=code))
        path=ROOT/'artifacts/queues/confirmation_pair_result.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(schema='aictrack1.confirmation.pair.v1',selected_arm=payload['selected_arm'],fresh_control=True,results=results,epoch=time.time()),indent=2)+'\n')
        if code:return code
    return 0
if __name__=='__main__':raise SystemExit(main())

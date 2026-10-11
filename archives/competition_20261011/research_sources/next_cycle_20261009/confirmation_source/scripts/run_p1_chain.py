"""Bounded two-arm controller. Each child uses the original canonical serial queue."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from p1_contract import ROOT,verify_files

def main():
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',choices=['J5','J20'],default=['J5','J20'])
    p.add_argument('--resume-arms',nargs='*',choices=['J5','J20'],default=[])
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--initial-free-bytes',type=int,required=True)
    p.add_argument('--disk-max-bytes',type=int,default=20*1024**3);p.add_argument('--min-free-bytes',type=int,default=3*1024**3)
    args=p.parse_args();verify_files()
    if len(args.arms)>2 or len(set(args.arms))!=len(args.arms):p.error('at most two distinct joint arms')
    results=[]
    for arm in args.arms:
        command=[sys.executable,'scripts/run_p1.py','--arm',arm,'--deadline-unix',str(args.deadline_unix),
            '--initial-free-bytes',str(args.initial_free_bytes),'--disk-max-bytes',str(args.disk_max_bytes),
            '--min-free-bytes',str(args.min_free_bytes)]
        if arm in args.resume_arms:command.append('--resume')
        code=subprocess.call(command,cwd=ROOT)
        results.append(dict(arm=arm,return_code=code))
        output=ROOT/'artifacts/queues/p1_chain_result.json';output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(dict(schema='aictrack1.p1.chain.v1',results=results,epoch=time.time()),indent=2)+'\n')
        if code:return code
    return 0
if __name__=='__main__':raise SystemExit(main())

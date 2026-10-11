"""One-byte GO before unchanged trainer import; exec retains PID/session/parent."""
import os,sys
from pathlib import Path
from p1_contract import ROOT,REMOTE_ROOT,ARMS,require

def main():
    require(os.name=='posix' and ROOT.resolve()==REMOTE_ROOT.resolve() and Path.cwd().resolve()==ROOT,'dedicated joint root/cwd required')
    arm=os.environ.get('P1_ARM');require(arm in ARMS,'registered arm required')
    expected=[sys.executable,'scripts/train_n1_peft.py','--config',f'configs/experiment/p1_{arm}.yaml']
    if os.environ.get('P1_RESUME')=='1':expected+=['--resume',str(ROOT/f'artifacts/experiments/JOINT_{arm}/last.pt')]
    if os.environ.get('P1_SMOKE')=='1':expected+=['--p1-smoke']
    if os.environ.get('P1_PROFILE')=='1':expected+=['--p1-profile']
    require(sys.argv[1:2]==['--'] and sys.argv[2:]==expected,'only exact unchanged trainer command allowed')
    if sys.stdin.buffer.read(1)!=b'G':return 73
    os.execvpe(expected[0],expected,os.environ.copy())
    raise RuntimeError('exec returned')
if __name__=='__main__':raise SystemExit(main())

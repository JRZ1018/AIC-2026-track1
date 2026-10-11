"""Exact H1 command receives GO only after canonical trainer attachment."""
import os,sys
from pathlib import Path
from confirmation_contract import ROOT,DEPLOYMENT_ROOT,require

def main():
    require(os.name=='posix' and ROOT.resolve()==DEPLOYMENT_ROOT.resolve() and Path.cwd().resolve()==ROOT,'dedicated H1 ROOT/cwd required')
    expected=[sys.executable,'scripts/train_frozen_head.py','--config','configs/experiment/confirmation_H1.yaml']
    if os.environ.get('P1_RESUME')=='1':expected+=['--resume',str(ROOT/'artifacts/experiments/confirmation_H1/last.pt')]
    require(sys.argv[1:2]==['--'] and sys.argv[2:]==expected,'only exact H1 trainer command allowed')
    if sys.stdin.buffer.read(1)!=b'G':return 73
    os.execvpe(expected[0],expected,os.environ.copy())
    raise RuntimeError('exec returned')
if __name__=='__main__':raise SystemExit(main())

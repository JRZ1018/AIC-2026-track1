"""Bind real heldout features/logits/source hashes; no graph or model inference."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,json
from pathlib import Path
from p3_contract import bind,verify_source

def main():
    p=argparse.ArgumentParser()
    for name in ('features','logits','identity','checkpoint','config','training-source','diagnostic-source','query-directory','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify_source()
    import torch
    torch.set_num_threads(2)
    print(json.dumps(bind(a)));return 0
if __name__=='__main__':raise SystemExit(main())

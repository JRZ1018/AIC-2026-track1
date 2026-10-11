"""Freeze fresh support-only queries before C/J predictions exist; CPU only."""
import argparse,hashlib
from pathlib import Path
import numpy as np
from confirmation_diag_contract import ROOT,confirmation_rows
from compare_confirmation import prepare_queries
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'inputs/frozen_queries');args=p.parse_args()
 _,parts=confirmation_rows();rows=parts['validation'];ids=[r['sample_id'] for r in rows]
 control=dict(ids=ids,groups=[r['group'] for r in rows],targets=np.array([int(r['target']) for r in rows]),
  # Only the number of class columns is observed; no logits, predictions or checkpoint is read.
  scores=np.empty((0,750)),identity=dict(sample_ids_order_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest()))
 result=prepare_queries(control,args.output)
 print({p['name']:(p['support']['rows'],len(p['columns'])) for p in result['protocols']})
if __name__=='__main__':main()

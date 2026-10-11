"""Exact legal full H1 identity plus the original full100 positive row contract."""
from pathlib import Path
from p1_contract import assets,sha,FULL_H1_SHA
from full_h1_reference import load_new_fold_head as strict_full_head

def load_new_fold_head(path,**kwargs):
    payload=assets();item=payload['assets']['h1']
    if Path(path).resolve()!=Path(item['path']).resolve() or sha(path)!=FULL_H1_SHA:raise ValueError('full H1 exact identity mismatch')
    if kwargs.get('forbid_full_train_initializer') is True:raise ValueError('full H1 forbidden for heldout caller')
    head,state=strict_full_head(path,**kwargs)
    if state['row_binding']['counts']!={'train':148695,'validation':0}:raise ValueError('positive full H1 count required')
    return head,state

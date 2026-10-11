"""The already selected graph point; unlabeled scores/features, no model call."""
from __future__ import annotations
import math,time
import numpy as np
import torch
from p3_graph import blocked_topk,mutual_graph,smooth,p7,check

PARAMETERS={'k':10,'lambda':.3,'iterations':3,'probability_temperature':1.,'protection_raw_margin':math.log(4.)}
SOURCE_SHA='5748cf07c01bd8bf3d68bf5b50065303d2c8990260b00400e5511593aaa0834c'
PROTOCOL_SHA='f48211db3716faaf45255dcc96e0b614bf42c1673791406bf9d7457c1a50b30d'

def apply_fixed_graph(scores,features,*,deadline,parameters=None):
    if parameters is not None and parameters!=PARAMETERS:raise ValueError('only selected k10/lambda.3 graph point')
    if deadline is None or not math.isfinite(deadline):raise ValueError('concrete final graph deadline required')
    torch.set_num_threads(2)
    values=np.asarray(scores,dtype=np.float32);embeddings=np.asarray(features,dtype=np.float32)
    if values.ndim!=2 or embeddings.shape!=(len(values),512) or values.shape[1]!=750:raise ValueError('same checkpoint750/512 row binding required')
    started=time.monotonic();check(deadline)
    neighbors,weights,top=blocked_topk(embeddings,20,row_block=256,max_block_bytes=64*1024**2,deadline=deadline)
    transition,graph,degree=mutual_graph(neighbors,weights,10)
    transformed,smoothing=smooth(values,transition,degree,.3,iterations=3,deadline=deadline)
    predictions,decoder=p7(transformed,deadline=deadline)
    check(deadline)
    return predictions,transformed,dict(parameters=PARAMETERS,scientific_source_sha256=SOURCE_SHA,
        upstream_protocol_sha256=PROTOCOL_SHA,topk=top,graph=graph,smoothing=smoothing,p7=decoder,
        wall_seconds=time.monotonic()-started,cpu_threads=2,model_updates=0,additional_model_forwards=0,
        test_labels_used=False,extra_label_anchors=0,parameter_search=False,
        budget_scope='independent final graph phase; original confirmation ledger untouched')

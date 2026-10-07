import pytest
import torch

from aictrack1.methods.local_group_support import local_group_support
from aictrack1.methods.local_group_support import independent_queries


def test_independent_queries_skip_previous_and_no_replacement():
    import hashlib
    ids=['a','b','c','d','e','f'];labels=[0,0,0,1,1,2]
    q=independent_queries(ids,labels,num_classes=3)
    first={min([i for i,l in enumerate(labels) if l==c],key=lambda i:hashlib.sha256(f'20261003:{ids[i]}'.encode()).digest()) for c in range(3)}
    assert len(q)==3 and len(set(q))==3 and not set(q)&first
    assert independent_queries(ids,labels,num_classes=3)==q


def test_independent_queries_reject_duplicate_ids():
    with pytest.raises(ValueError):independent_queries(['a','a'],[0,1],num_classes=2)


def test_excludes_whole_group_not_only_query():
    x = torch.tensor([[1.,0.],[1.,0.],[.9,.1],[0.,1.]])
    r = local_group_support(x,torch.tensor([0,0,1,0]),['a','a','b','c'],[0],num_classes=2,k=2)
    assert r['neighbour_components'] == [[1,2]]
    assert torch.equal(r['target_support'],torch.tensor([.5]))


def test_one_vote_per_group_with_conflicting_labels():
    x = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
    r = local_group_support(x,torch.tensor([0,0,0,1,1]),['q','g','g','g','z'],[0],num_classes=2,k=1)
    assert abs(float(r['target_support'][0])-2/3)<1e-6


def test_label_blind_ties_and_chunk_equivalence():
    x=torch.ones(4,2); y=torch.tensor([0,1,0,1]); g=['z','a','b','c']
    a=local_group_support(x,y,g,[0,1],num_classes=2,k=2,query_batch=1)
    b=local_group_support(x,y,g,[0,1],num_classes=2,k=2,query_batch=2)
    assert a['neighbour_components']==b['neighbour_components']==[[0,1],[1,2]]
    assert torch.equal(a['target_support'],b['target_support'])


def test_guard_every_chunk_no_gradient():
    calls=[]; x=torch.eye(4,requires_grad=True)
    r=local_group_support(x,torch.tensor([0,0,1,1]),list('abcd'),[0,1,2],num_classes=2,k=1,query_batch=2,guard=lambda:calls.append(1))
    assert len(calls)==2 and not r['target_support'].requires_grad and x.grad is None


def test_audit_terminal_uses_canonical_output_schema(tmp_path):
    import json
    from aictrack1.engine.training_queue import inspect_run_output
    output=tmp_path/'audit';output.mkdir()
    (output/'query_support.pt').write_bytes(b'safe scalar payload fixture, not a model')
    assert inspect_run_output(tmp_path,'audit').status=='partial'
    (output/'config.yaml').write_text('{}')
    (output/'command.txt').write_text('[]')
    (output/'metrics.json').write_text(json.dumps(dict(selected_checkpoint='query_support.pt',role='audit_scalar_payload_not_a_trained_model')))
    assert inspect_run_output(tmp_path,'audit').status=='complete'


@pytest.mark.parametrize('kind',['nan','zero','labels','groups','queries','k'])
def test_bad_contract(kind):
    x=torch.eye(3); y=torch.tensor([0,1,0]); g=list('abc'); q=[0]; k=1
    if kind=='nan':x[0,0]=float('nan')
    if kind=='zero':x[0]=0
    if kind=='labels':y[1]=2
    if kind=='groups':g[1]=''
    if kind=='queries':q=[0,0]
    if kind=='k':k=3
    with pytest.raises(ValueError):local_group_support(x,y,g,q,num_classes=2,k=k)

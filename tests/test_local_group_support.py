import pytest
import torch

from aictrack1.methods.local_group_support import local_group_support


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

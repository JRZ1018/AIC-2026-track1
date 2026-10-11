"""Independent CPU reciprocal-cosine probability smoothing; no model or labels."""
from __future__ import annotations
import math,time
import numpy as np
import torch
from scipy.sparse import csr_matrix

MARGIN=math.log(4.)
GRID=((10,.1),(10,.3),(20,.1),(20,.3))

class BudgetExceeded(RuntimeError):pass

def check(deadline):
    if deadline is not None and time.monotonic()>=deadline:raise BudgetExceeded('aggregate P3 wall budget exhausted')

def finite_matrix(values,name):
    array=np.asarray(values)
    if array.ndim!=2 or array.shape[0]<1 or array.shape[1]<1 or not np.isfinite(array).all():
        raise ValueError(name+' must be finite nonempty [N,D]')
    return np.asarray(array,dtype=np.float32)

def blocked_topk(features,max_k=20,*,row_block=256,max_block_bytes=64*1024**2,deadline=None):
    """Only BxN similarity blocks; ties use original query row, without epsilon."""
    values=finite_matrix(features,'features');n,d=values.shape
    if max_k<1 or row_block<1 or max_block_bytes<4*n:raise ValueError('invalid bounded top-k settings')
    norms=np.linalg.norm(values,axis=1);valid=norms>0
    if np.any(valid & (np.abs(norms-1)>2e-5)):raise ValueError('features must already be normalized; zero rows become isolated')
    k=min(max_k,max(0,int(valid.sum())-1));neighbors=np.full((n,k),-1,np.int64);weights=np.zeros((n,k),np.float32)
    if not k:return neighbors,weights,dict(rows=n,feature_dim=d,max_k=max_k,effective_k=0,
        zero_feature_rows=int((~valid).sum()),similarity_block_rows=0,peak_similarity_block_bytes=0,
        tie_cutoff_rows=0,dense_N_by_N_allocated=False)
    tensor=torch.from_numpy(np.ascontiguousarray(values));block=min(row_block,max(1,max_block_bytes//(4*n)))
    # Even a small fixture must not silently allocate an NxN score matrix.
    if n>1:block=min(block,n-1)
    peak=0;ties=0
    for start in range(0,n,block):
        check(deadline);stop=min(start+block,n)
        sims=(tensor[start:stop]@tensor.T).numpy();peak=max(peak,sims.nbytes)
        sims[:,~valid]=-np.inf
        for local,row in enumerate(range(start,stop)):
            sims[local,row]=-np.inf
            if not valid[row] or not k:continue
            line=sims[local]
            # partition finds the cutoff; all cutoff ties are then resolved by row order.
            indices=np.argpartition(-line,k-1)[:k];cutoff=line[indices].min()
            better=np.flatnonzero(line>cutoff);equal=np.flatnonzero(line==cutoff)
            ties+=int(len(equal)>k-len(better))
            selected=np.r_[better,equal[:k-len(better)]]
            selected=selected[np.lexsort((selected,-line[selected]))]
            neighbors[row]=selected;weights[row]=line[selected]
        del sims
    return neighbors,weights,dict(rows=n,feature_dim=d,max_k=max_k,effective_k=k,zero_feature_rows=int((~valid).sum()),
        similarity_block_rows=block,peak_similarity_block_bytes=peak,tie_cutoff_rows=ties,dense_N_by_N_allocated=False)

def mutual_graph(neighbors,weights,k):
    """Symmetric unnormalized positive reciprocal weights, row-stochastic W."""
    if neighbors.shape!=weights.shape or neighbors.ndim!=2 or k<1:raise ValueError('invalid directed neighbors')
    n=len(neighbors);width=min(k,neighbors.shape[1]);ns=neighbors[:,:width];ws=weights[:,:width]
    rows=np.repeat(np.arange(n,dtype=np.int64),width);cols=ns.reshape(-1);data=ws.reshape(-1)
    valid=(cols>=0)&(cols!=rows)&(data>0)&np.isfinite(data)
    rows,cols,data=rows[valid],cols[valid],data[valid]
    keys=rows*n+cols;order=np.argsort(keys);keys,rows,cols,data=keys[order],rows[order],cols[order],data[order]
    reciprocal=cols*n+rows;where=np.searchsorted(keys,reciprocal)
    keep=np.zeros(len(keys),bool);within=where<len(keys);keep[within]=keys[where[within]]==reciprocal[within]
    asym=float(np.abs(data[keep]-data[where[keep]]).max(initial=0))
    average=(data[keep]+data[where[keep]])*.5
    adjacency=csr_matrix((average,(rows[keep],cols[keep])),shape=(n,n),dtype=np.float32)
    adjacency.sort_indices();mass=np.asarray(adjacency.sum(1)).reshape(-1);degree=np.diff(adjacency.indptr)
    inv=np.divide(1.,mass,out=np.zeros_like(mass),where=mass>0)
    transition=adjacency.multiply(inv[:,None]).tocsr();transition.sort_indices()
    if (adjacency!=adjacency.T).nnz or adjacency.diagonal().any():raise RuntimeError('mutual graph is not symmetric without self edges')
    sums=np.asarray(transition.sum(1)).reshape(-1)
    if not np.allclose(sums[degree>0],1,atol=2e-6,rtol=0):raise RuntimeError('transition mass differs')
    return transition,dict(k=k,directed_positive_edges=len(keys),directed_nonpositive_or_invalid_pruned=int(n*width-len(keys)),
        reciprocal_directed_edges=adjacency.nnz,undirected_edges=adjacency.nnz//2,
        zero_degree_rows=int((degree==0).sum()),min_degree=int(degree.min()),max_degree=int(degree.max()),
        mean_degree=float(degree.mean()),degree_histogram=np.bincount(degree,minlength=k+1).tolist(),
        reciprocal_weight_asymmetry_before_average=asym,symmetric_adjacency=True,self_edges=0),degree

def smooth(scores,transition,degree,mixing,*,iterations=3,deadline=None):
    values=finite_matrix(scores,'scores')
    if values.shape[1]<2 or transition.shape!=(len(values),len(values)) or len(degree)!=len(values):raise ValueError('score/graph shape mismatch')
    if mixing not in (.0,.1,.3) or iterations!=3:raise ValueError('three residual iterations and registered lambda required')
    check(deadline)
    tensor=torch.from_numpy(values);logp=torch.log_softmax(tensor,dim=1).numpy();base=np.exp(logp).astype(np.float32)
    logz=torch.logsumexp(tensor,dim=1).numpy()
    # Stable row-index class tie break is the default argmax, not a label lookup.
    top=np.partition(values,-2,axis=1)[:,-2:];margin=top.max(1)-top.min(1)
    protected=margin>=MARGIN;isolated=np.asarray(degree)==0;anchor=protected|isolated
    q=base.copy()
    for _ in range(3):
        check(deadline);next_q=(1-mixing)*base+mixing*(transition@q)
        next_q[anchor]=base[anchor];q=np.asarray(next_q,dtype=np.float32)
    if not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=3e-6,rtol=0):raise RuntimeError('smoothed probability simplex failure')
    tiny=np.finfo(np.float32).tiny
    transformed=np.log(np.maximum(q,tiny))+logz[:,None]
    unchanged=np.all(q==base,axis=1);preserve=anchor|unchanged
    transformed[preserve]=values[preserve]  # exact logits and all margins, including numerical zeros.
    if not np.array_equal(transformed[anchor],values[anchor]):raise RuntimeError('margin/zero-degree protection failure')
    changed=transformed.argmax(1)!=values.argmax(1)
    top_out=np.partition(transformed,-2,axis=1)[:,-2:];new_margin=top_out.max(1)-top_out.min(1)
    return transformed.astype(np.float32),dict(lambda_value=mixing,iterations=3,original_probability_temperature=1.,
        residual_anchor='original P at every iteration',protected_margin_threshold=MARGIN,protected_rows=int(protected.sum()),
        zero_degree_rows=int(isolated.sum()),protected_or_zero_degree_rows=int(anchor.sum()),unchanged_probability_rows=int(unchanged.sum()),
        protected_logit_rows_exactly_preserved=True,protected_raw_prediction_changes=int(changed[protected].sum()),
        isolated_raw_prediction_changes=int(changed[isolated].sum()),prediction_changes=int(changed.sum()),
        initial_zero_probability_entries=int((base==0).sum()),final_zero_probability_entries=int((q==0).sum()),
        final_probability_floor_entries=int((q<tiny).sum()),probability_floor=float(tiny),
        margin_change_abs_max_protected=float(np.abs(new_margin[protected]-margin[protected]).max(initial=0)),
        margin_change_abs_max_all=float(np.abs(new_margin-margin).max(initial=0)),probability_mass_error_max=float(np.abs(q.sum(1)-1).max()))

def p7(scores,*,deadline=None):
    """Exactly the existing P7 .5/.75/60; graph probabilities used T=1 only."""
    values=torch.from_numpy(finite_matrix(scores,'P7 scores'));top=torch.topk(values,2,dim=1).values
    median=float((top[:,0]-top[:,1]).median());scale=max(median*.5,1e-4)
    normalized=values/scale;bias=torch.zeros(values.shape[1],dtype=torch.float32)
    for iteration in range(1,61):
        check(deadline);mass=torch.softmax(normalized+bias,1).mean(0).clamp_min(1e-12)
        correction=-math.log(values.shape[1])-mass.log();bias+=correction;bias-=bias.mean()
        if float(correction.abs().max())<=1e-4:break
    adjusted=values+.75*(bias*scale)
    return adjusted.argmax(1).numpy(),dict(name='P7',temperature=.5,strength=.75,max_iterations=60,
        actual_iterations=iteration,temperature_scale=scale,median_margin=median)

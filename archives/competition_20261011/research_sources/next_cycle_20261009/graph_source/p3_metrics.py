"""Evaluation only, deliberately separate from the label-free graph module."""
import numpy as np

def paired(prediction,baseline,targets,groups,train_counts,protected,*,draws=500,seed=20261009):
    prediction=np.asarray(prediction);baseline=np.asarray(baseline);targets=np.asarray(targets)
    hit=prediction==targets;old=baseline==targets;delta=hit.astype(np.float64)-old
    unique,inverse=np.unique(np.asarray(groups),return_inverse=True)
    sums=np.bincount(inverse,weights=delta,minlength=len(unique));sizes=np.bincount(inverse,minlength=len(unique))
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        selected=rng.integers(0,len(unique),size=len(unique));samples.append(100*sums[selected].sum()/sizes[selected].sum())
    ci=np.percentile(samples,[2.5,97.5]);classes=len(train_counts)
    support=np.bincount(targets,minlength=classes);correct=np.bincount(targets,weights=hit,minlength=classes)
    old_correct=np.bincount(targets,weights=old,minlength=classes);recall=np.divide(correct,support,out=np.zeros(classes),where=support>0)
    ranks=np.argsort(train_counts,kind='stable');strata={}
    for name,indices in zip(('tail','mid','head'),np.array_split(ranks,3)):
        rows=np.isin(targets,indices);n=int(rows.sum())
        strata[name]=dict(samples=n,classes=len(indices),observed_accuracy=float(hit[rows].mean()) if n else None,
            delta_pp=float(100*delta[rows].mean()) if n else None)
    changed=prediction!=baseline
    return dict(observed_accuracy=float(hit.mean()),baseline_observed_accuracy=float(old.mean()),delta_pp=float(100*delta.mean()),
        right_to_wrong=int((old&~hit).sum()),wrong_to_right=int((hit&~old).sum()),prediction_changes=int(changed.sum()),
        protected_prediction_changes=int(changed[protected].sum()),protected_right_to_wrong=int((old&~hit&protected).sum()),
        protected_wrong_to_right=int((hit&~old&protected).sum()),macro_observed_accuracy=float(recall[support>0].mean()),
        support=support.tolist(),per_class_correct=correct.astype(int).tolist(),per_class_baseline_correct=old_correct.astype(int).tolist(),
        unique_group_count=len(unique),zero_recall_class_count=int(((correct==0)&(support>0)).sum()),strata=strata,
        group_bootstrap_95ci_pp=ci.tolist(),ci_crosses_zero=bool(ci[0]<=0<=ci[1]),stable_gain_claim=False)

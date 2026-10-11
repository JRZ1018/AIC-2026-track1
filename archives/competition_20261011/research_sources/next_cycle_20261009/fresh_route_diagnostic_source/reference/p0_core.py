"""Original frozen-score P0 audit, no images/models/optimizer or external data.

Soft rule preserves this project's P7 formula. Capacity optimization is a sparse
transport LP (equivalent min-cost flow), with integer network capacities. HiGHS
dual simplex provides an integral basic optimum on the supplied top-k graph.
It is NOT a claim of dense global optimality. SciPy is the only solver dependency.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack


class InfeasibleGraph(RuntimeError):
    pass


class SolverTimeout(RuntimeError):
    pass


def validate_scores(scores):
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError('scores must be nonempty [N,C], C>=2')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite scores')
    return values


def soft_bias(scores, temperature=0.5, iterations=60):
    """P7-compatible CPU FP32 bias, relative to lower median top-two margin."""
    values = torch.from_numpy(validate_scores(scores))
    if not math.isfinite(temperature) or temperature <= 0 or iterations < 1:
        raise ValueError('invalid soft settings')
    top2 = torch.topk(values, k=2, dim=1).values
    median = float((top2[:, 0] - top2[:, 1]).median())
    scale = max(median * temperature, 1e-4)
    normalized = values / scale
    bias = torch.zeros(values.shape[1], dtype=torch.float32)
    for iteration in range(1, iterations + 1):
        mass = torch.softmax(normalized + bias, dim=1).mean(0).clamp_min(1e-12)
        correction = -math.log(values.shape[1]) - mass.log()
        bias += correction
        bias -= bias.mean()
        error = float(correction.abs().max())
        if error <= 1e-4:
            break
    return (bias * scale).numpy(), dict(median_top1_top2_margin=median,
            temperature_scale=scale, iterations=iteration, max_log_mass_error=error,
            calculation_dtype='float32', target='uniform')


def capacity_bounds(n, c, ratios):
    lower, upper = ratios
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= 1 <= upper):
        raise ValueError('capacity ratios must contain one')
    # Integer envelope, not rounding to a falsely exact 49/50 quota.
    return math.floor(lower * n / c), math.ceil(upper * n / c)


def solve_sparse_capacity(scores, lower, upper, top_k, time_limit=3600):
    started = time.monotonic()
    values = validate_scores(scores)
    n, c = values.shape
    if not isinstance(lower, int) or not isinstance(upper, int):
        raise ValueError('capacities must be integers')
    if lower < 0 or upper < lower or lower * c > n or upper * c < n:
        raise ValueError('infeasible capacity bounds')
    if not 1 <= top_k <= c or time_limit <= 0:
        raise ValueError('invalid top-k/time budget')
    # Stable class-index tie break. No expanded N x N class-slot matrix.
    edges = np.argsort(-values, axis=1, kind='stable')[:, :top_k].copy()
    rows = np.repeat(np.arange(n), top_k)
    classes = edges.reshape(-1)
    variables = np.arange(rows.size)
    ones = np.ones(rows.size)
    row_constraints = coo_matrix((ones, (rows, variables)), shape=(n, rows.size)).tocsr()
    class_constraints = coo_matrix((ones, (classes, variables)), shape=(c, rows.size)).tocsr()
    upper_constraints = vstack([class_constraints, -class_constraints], format='csr')
    rhs = np.r_[np.full(c, upper), np.full(c, -lower)]
    # Float64 subtraction of FP32 stored scores avoids cancellation in regret.
    edge_scores = values[rows, classes].astype(np.float64)
    regrets = values.max(axis=1).astype(np.float64)[rows] - edge_scores
    remaining = time_limit - (time.monotonic() - started)
    if remaining <= 0:
        raise SolverTimeout('preparation exhausted time budget')
    with warnings.catch_warnings():
        # SciPy forwards this recognized HiGHS setting although wrapper may warn.
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
        result = linprog(regrets, A_ub=upper_constraints, b_ub=rhs,
                         A_eq=row_constraints, b_eq=np.ones(n), bounds=(0, 1),
                         method='highs-ds', options={'time_limit': remaining,
                         'threads': 1, 'primal_feasibility_tolerance': 1e-8,
                         'dual_feasibility_tolerance': 1e-8})
    elapsed = time.monotonic() - started
    if result.status == 2:
        raise InfeasibleGraph(f'top-{top_k} graph infeasible: {result.message}')
    if result.status == 1:
        raise SolverTimeout(f'no certified optimum within budget: {result.message}')
    if not result.success:
        raise RuntimeError(f'HiGHS status {result.status}: {result.message}')
    x = result.x.reshape(n, top_k)
    integral_error = float(np.abs(x - np.rint(x)).max())
    if integral_error > 1e-6 or not np.allclose(x.sum(axis=1), 1, atol=1e-7, rtol=0):
        raise RuntimeError('solver did not return an integral one-assignment solution')
    predictions = edges[np.arange(n), x.argmax(axis=1)].astype(np.int64)
    counts = np.bincount(predictions, minlength=c)
    if np.any(counts < lower) or np.any(counts > upper):
        raise RuntimeError('decoded assignment violates capacity bounds')
    assigned_regret = float((values.max(axis=1).astype(np.float64)
                    - values[np.arange(n), predictions].astype(np.float64)).sum())
    gap = abs(assigned_regret - float(result.fun))
    tolerance = max(1e-5, 1e-8 * max(1.0, abs(float(result.fun))))
    if gap > tolerance:
        raise RuntimeError('integer decode and LP optimum disagree')
    return predictions, dict(method='sparse_transport_lp_highs_dual_simplex',
            optimality_scope='supplied top-k graph only; no dense certificate',
            solver_status=int(result.status), top_k=top_k, edge_count=int(rows.size),
            lower=lower, upper=upper, elapsed_seconds=elapsed,
            primal_integrality_max_error=integral_error, lp_minimum_regret=float(result.fun),
            decoded_regret=assigned_regret, lp_decode_objective_gap=gap,
            raw_score_sum=float(values[np.arange(n), predictions].astype(np.float64).sum()),
            mean_regret=assigned_regret / n, counts=counts.tolist(),
            deterministic_settings='stable class ties; HiGHS threads=1',
            solver_message=str(result.message))


def decode_capacity(scores, ratios=(0.8, 1.2), top_ks=(5, 20), time_limit=3600):
    started = time.monotonic()
    values = validate_scores(scores)
    lower, upper = capacity_bounds(*values.shape, ratios)
    attempts = []
    ks = list(dict.fromkeys(min(int(k), values.shape[1]) for k in top_ks))
    for k in ks:
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            raise SolverTimeout('aggregate top-k budget exhausted')
        try:
            predictions, report = solve_sparse_capacity(values, lower, upper, k, remaining)
            report.update(attempts=attempts, ratios=list(ratios),
                          total_elapsed_seconds=time.monotonic() - started)
            return predictions, report
        except InfeasibleGraph as exc:
            attempts.append(dict(top_k=k, status='infeasible', message=str(exc)))
    raise InfeasibleGraph(f'no feasible assignment in registered graphs: {attempts}')


def unique_balanced_indices(targets, groups, per_class=20, seed=20261009):
    """Closed-set stress subtask, no row replacement, at most one per group.

    Inclusion uses observed-label/group support BEFORE candidate predictions.
    Shared conflicting groups may make a class ineligible. Ascending support
    priority is fixed; omitted classes are always disclosed.
    """
    targets = np.asarray(targets, dtype=np.int64)
    if len(groups) != len(targets) or per_class < 1:
        raise ValueError('invalid sampling input')
    rng = np.random.default_rng(seed)
    candidates = []
    for label in np.unique(targets):
        members = np.flatnonzero(targets == label)
        support = len({groups[i] for i in members})
        if support >= per_class:
            candidates.append((support, int(label), members))
    used = set()
    selected, retained = [], []
    for _, label, members in sorted(candidates, key=lambda item: item[:2]):
        proposed, current_groups = [], set()
        for i in rng.permutation(members):
            group = groups[int(i)]
            if group not in used and group not in current_groups:
                proposed.append(int(i))
                current_groups.add(group)
                if len(proposed) == per_class:
                    break
        if len(proposed) == per_class:
            selected.extend(proposed)
            retained.append(label)
            used.update(current_groups)
    if len(retained) < 2:
        raise ValueError('fewer than two classes have sufficient disjoint unique support')
    return np.asarray(selected, dtype=np.int64), np.asarray(sorted(retained), dtype=np.int64)


def support_report(targets, groups, sample_ids, num_classes):
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) != len(groups) or len(targets) != len(sample_ids):
        raise ValueError('support rows differ')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('duplicated sample IDs are not unique query support')
    return dict(rows=len(targets), unique_images=len(set(sample_ids)),
                unique_groups=len(set(groups)),
                observed_label_counts=np.bincount(targets, minlength=num_classes).tolist(),
                unique_groups_per_class=[len({groups[i] for i in np.flatnonzero(targets == c)})
                                        for c in range(num_classes)])


def paired_metrics(predictions, baseline, targets, groups, scores, draws=500, seed=20261009):
    predictions = np.asarray(predictions, dtype=np.int64)
    baseline = np.asarray(baseline, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    n, c = scores.shape
    if any(len(x) != n for x in (predictions, baseline, targets, groups)):
        raise ValueError('metric row mismatch')
    if any(((x < 0) | (x >= c)).any() for x in (predictions, baseline, targets)):
        raise ValueError('out-of-range labels/predictions')
    correct, control = predictions == targets, baseline == targets
    difference = correct.astype(np.int64) - control.astype(np.int64)
    support = np.bincount(targets, minlength=c)
    successes = np.bincount(targets, weights=correct, minlength=c)
    recall = np.divide(successes, support, out=np.full(c, np.nan), where=support > 0)
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    margins = top2.max(1) - top2.min(1)
    protected = margins >= np.quantile(margins, 0.9)
    _, codes = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    count = int(codes.max()) + 1
    group_n = np.bincount(codes, minlength=count)
    group_delta = np.bincount(codes, weights=difference, minlength=count)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(draws):
        selected = rng.integers(0, count, size=count)
        bootstrap.append(float(group_delta[selected].sum() / group_n[selected].sum()) * 100)
    return dict(metric='observed-label agreement; no hidden test labels',
                top1_pp=float(correct.mean() * 100), macro_pp=float(np.nanmean(recall) * 100),
                delta_vs_baseline_pp=float(difference.mean() * 100),
                correct_to_wrong=int((control & ~correct).sum()),
                wrong_to_correct=int((~control & correct).sum()),
                changed_predictions=int((baseline != predictions).sum()),
                high_margin_correct_to_wrong=int((protected & control & ~correct).sum()),
                high_margin_wrong_to_correct=int((protected & ~control & correct).sum()),
                high_margin_rule='raw top-two margin >= fixed query 90th percentile',
                mean_raw_regret=float((scores.max(1).astype(np.float64)
                           - scores[np.arange(n), predictions].astype(np.float64)).mean()),
                prediction_counts=np.bincount(predictions, minlength=c).tolist(),
                per_class_recall=[None if np.isnan(x) else float(x) for x in recall],
                group_bootstrap=dict(draws=draws, seed=seed, groups=count,
                     ci95_pp=np.quantile(bootstrap, [0.025, 0.975]).tolist() if draws else None,
                     limitation='conditional fixed-model observed labels; not training-seed variance'))

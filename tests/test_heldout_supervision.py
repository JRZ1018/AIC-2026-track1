import pytest
from aictrack1.methods.heldout_supervision import grouped_folds, exclusion_contract


def row(sid, label=0, **kw):
    return dict(sample_id=sid, internal_class_index=label, partition='train', source_split='train', **kw)


def test_transitive_exact_and_near_union():
    rows = [row('a', sha256='x'), row('b', sha256='x', near_duplicate_group='n'),
            row('c', near_duplicate_group='n'), row('d', duplicate_group='e'), row('z', duplicate_group='e')]
    folds, groups = grouped_folds(rows)
    assert groups[:3] == ['a'] * 3 and groups[3:] == ['d'] * 2
    assert len(set(folds[:3])) == len(set(folds[3:])) == 1


def test_order_and_label_blind_determinism():
    rows = [row(str(i), i % 2) for i in range(25)]
    a, _ = grouped_folds(rows)
    b, _ = grouped_folds([dict(r, internal_class_index=9) for r in reversed(rows)])
    assert a == b[::-1]


@pytest.mark.parametrize('field,value', [('partition', 'validation'), ('source_split', 'test')])
def test_other_partition_rejected(field, value):
    with pytest.raises(ValueError):
        grouped_folds([dict(row('a'), **{field: value})])


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        grouped_folds([row('a'), row('a')])


def test_exact_fit_exclusion_and_coverage():
    rows = [row('a', 0), row('b', 1), row('c', 0), row('d', 1)]
    fit, held, contract = exclusion_contract(rows, [0, 0, 1, 1], ['a', 'b', 'c', 'd'], heldout_fold=0, num_classes=2)
    assert fit == [2, 3] and held == [0, 1]
    assert contract['fit_class_counts'] == contract['heldout_class_counts'] == [1, 1]
    assert contract['group_overlap'] == 0


def test_group_leakage_rejected():
    with pytest.raises(ValueError, match='leaked'):
        exclusion_contract([row('a'), row('b')], [0, 1], ['same', 'same'], heldout_fold=0, num_classes=1)


def test_singleton_class_abstains_not_fallback_to_held_label():
    with pytest.raises(ValueError, match='abstain'):
        exclusion_contract([row('a', 0), row('b', 1)], [0, 1], ['a', 'b'], heldout_fold=0, num_classes=2)


def test_empty_fold_and_lengths_rejected():
    with pytest.raises(ValueError):
        exclusion_contract([row('a')], [0], ['a'], heldout_fold=1, num_classes=1)
    with pytest.raises(ValueError):
        exclusion_contract([row('a')], [], ['a'], heldout_fold=0, num_classes=1)


def test_forged_components_cannot_bypass_exact_exclusion():
    with pytest.raises(ValueError, match='despite'):
        exclusion_contract([row('a', sha256='same'), row('b', sha256='same')],
                           [0, 1], ['a', 'b'], heldout_fold=0, num_classes=1)


def test_fold_type_and_duplicate_ids_rejected():
    for rows, assignment in [([row('a'), row('b')], [0, -1]), ([row('a'), row('a')], [0, 1])]:
        with pytest.raises(ValueError):
            exclusion_contract(rows, assignment, ['a', 'b'], heldout_fold=0, num_classes=1)

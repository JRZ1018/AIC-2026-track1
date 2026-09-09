from __future__ import annotations

import torch

from aictrack1.methods.confidence import classwise_rank_noise_scores, select_suspected_noisy


def test_classwise_scores_and_tail_protection() -> None:
    losses = torch.tensor([0.1, 0.2, 2.0, 0.3, 0.4, 3.0, 5.0])
    classes = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    scores = classwise_rank_noise_scores(losses, classes)
    selected = select_suspected_noisy(scores, classes, fraction=1 / 3, min_keep_per_class=2)
    assert selected.tolist() == [False, False, True, False, False, True, False]
    assert not selected[classes == 2].any()


def test_tied_losses_receive_equal_scores() -> None:
    scores = classwise_rank_noise_scores(torch.ones(4), torch.tensor([0, 0, 0, 0]))
    assert torch.equal(scores, torch.full((4,), 0.5))

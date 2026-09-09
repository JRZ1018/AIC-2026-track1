from __future__ import annotations

import torch

from aictrack1.evaluation.p2_benchmark import (
    P2BenchmarkConfig,
    SyntheticNoiseTrainer,
    make_known_noise_data,
    run_known_noise_benchmark,
)


def _config(steps: int = 16) -> P2BenchmarkConfig:
    return P2BenchmarkConfig(
        seed=37,
        num_classes=4,
        samples_per_class=18,
        feature_dim=8,
        feature_noise=0.08,
        noise_rate=1 / 3,
        steps=steps,
        learning_rate=0.2,
        logit_tau=1.0,
        elr_lambda=0.25,
    )


def test_ce_robust_losses_and_confidence_behave_on_known_noise() -> None:
    result = run_known_noise_benchmark(_config())
    assert result["realized_noisy_count"] == 24
    for method in ("ce", "logitclip", "elr"):
        metrics = result["methods"][method]
        assert metrics["final_observed_ce"] < metrics["initial_observed_ce"]
        detection = metrics["noise_detection"]["overall"]
        assert detection["auroc"] > 0.8
        assert detection["precision"] > 0.6
    assert result["methods"]["logitclip"]["clipped_fraction"] > 0


def test_elr_checkpoint_resume_matches_uninterrupted_training(tmp_path) -> None:
    config = _config(steps=10)
    data = make_known_noise_data(config)
    uninterrupted = SyntheticNoiseTrainer(data, config, "elr")
    for _ in range(10):
        uninterrupted.step()

    interrupted = SyntheticNoiseTrainer(data, config, "elr")
    for _ in range(4):
        interrupted.step()
    checkpoint = tmp_path / "elr.pt"
    interrupted.save_checkpoint(checkpoint)
    resumed = SyntheticNoiseTrainer(data, config, "elr")
    resumed.load_checkpoint(checkpoint)
    for _ in range(6):
        resumed.step()

    assert resumed.step_number == uninterrupted.step_number == 10
    for left, right in zip(uninterrupted.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(left, right)
    assert uninterrupted.elr_state is not None and resumed.elr_state is not None
    assert torch.equal(uninterrupted.elr_state.target, resumed.elr_state.target)

import math

import pytest
import torch

from specdecode.rejection_sampling import (
    acceptance_prob,
    entropy_nats,
    logits_to_probs,
    residual_probs,
    sample_from_probs,
)


def test_logits_to_probs_returns_normalized_probabilities():
    logits = torch.tensor([1.0, 2.0, 3.0])
    probs = logits_to_probs(logits, temperature=1.0)

    assert probs.dtype == torch.float32
    assert torch.all(probs >= 0)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_logits_to_probs_respects_temperature():
    logits = torch.tensor([0.0, 2.0])
    cold = logits_to_probs(logits, temperature=0.5)
    hot = logits_to_probs(logits, temperature=2.0)

    assert cold[1] > hot[1]


def test_logits_to_probs_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        logits_to_probs(torch.tensor([1.0, 2.0]), temperature=0.0)

    with pytest.raises(ValueError):
        logits_to_probs(torch.tensor([1.0, 2.0]), temperature=-1.0)


def test_acceptance_probability_is_min_one_p_over_q():
    assert acceptance_prob(torch.tensor(0.2), torch.tensor(0.4)) == pytest.approx(0.5)
    assert acceptance_prob(torch.tensor(0.8), torch.tensor(0.4)) == pytest.approx(1.0)


def test_residual_probs_is_normalized_and_nonnegative():
    p = torch.tensor([0.1, 0.6, 0.3])
    q = torch.tensor([0.4, 0.2, 0.4])

    residual = residual_probs(p, q)

    assert torch.all(residual >= 0)
    assert torch.isclose(residual.sum(), torch.tensor(1.0))
    assert torch.allclose(residual, torch.tensor([0.0, 1.0, 0.0]))


def test_residual_probs_falls_back_to_p_when_mass_is_zero():
    p = torch.tensor([0.2, 0.3, 0.5])
    q = p.clone()

    residual = residual_probs(p, q)

    assert torch.allclose(residual, p)


def test_entropy_nats_for_certain_distribution_is_zero():
    probs = torch.tensor([1.0, 0.0, 0.0])

    assert entropy_nats(probs) == pytest.approx(0.0, abs=1e-7)


def test_entropy_nats_for_uniform_binary_distribution_is_log_two():
    probs = torch.tensor([0.5, 0.5])

    assert entropy_nats(probs) == pytest.approx(math.log(2.0), rel=1e-6)


def test_sample_from_probs_returns_only_nonzero_probability_token():
    probs = torch.tensor([0.0, 1.0, 0.0])

    for _ in range(10):
        assert sample_from_probs(probs) == 1

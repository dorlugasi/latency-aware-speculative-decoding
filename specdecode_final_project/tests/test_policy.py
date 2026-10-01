import pytest
import torch

from specdecode.policy import (
    LinUCBPolicy,
    acceptance_features,
    entropy_features,
    extract_acceptance_features,
    extract_combined_features,
    extract_entropy_features,
    extract_latency_features,
    latency_features,
)
from specdecode.speculative import GenerationState, StepResult


def make_step(
    step_index=0,
    k=4,
    accepted=2,
    output_tokens=3,
    draft_s=0.010,
    verify_s=0.008,
    wall_s=0.030,
    entropy_mean=1.0,
    entropy_max=1.5,
    entropy_last=1.2,
):
    return StepResult(
        step_index=step_index,
        k_requested=k,
        num_accepted=accepted,
        new_token_ids=[1] * output_tokens,
        draft_time_s=draft_s,
        verify_time_s=verify_s,
        step_time_s=wall_s,
        draft_entropy_mean=entropy_mean,
        draft_entropy_max=entropy_max,
        draft_entropy_last=entropy_last,
    )


def state(history=None):
    return GenerationState(
        step_index=len(history or []),
        generated_token_ids=[],
        max_new_tokens=1000,
        history=history or [],
    )


def test_latency_features_empty_history_are_zero():
    f = latency_features(state())
    assert f.values() == [0.0] * 7


def test_latency_signal_uses_model_compute_not_round_wall_time():
    """Changing Python/policy overhead must not change latency-policy features."""
    a = make_step(draft_s=0.010, verify_s=0.020, wall_s=0.035)
    b = make_step(draft_s=0.010, verify_s=0.020, wall_s=0.500)

    fa = extract_latency_features(state([a]))
    fb = extract_latency_features(state([b]))

    assert torch.equal(fa, fb)
    assert fa[2].item() == pytest.approx(30.0)  # draft + verify CUDA-event time


def test_latency_features_have_expected_normalized_values():
    s = make_step(k=5, output_tokens=4, draft_s=0.010, verify_s=0.020)
    f = latency_features(state([s]))
    assert f.prev_k == 5
    assert f.step_ms == pytest.approx(30.0)
    assert f.draft_ms_per_token == pytest.approx(2.0)
    assert f.verify_ms_per_token == pytest.approx(4.0)
    assert f.draft_verify_ratio == pytest.approx(0.5)
    assert f.output_ms_per_token == pytest.approx(7.5)


def test_entropy_features_use_mean_max_last_and_recent_mean():
    h = [
        make_step(entropy_mean=1.0, entropy_max=1.4, entropy_last=1.2),
        make_step(entropy_mean=2.0, entropy_max=2.7, entropy_last=2.3),
    ]
    f = entropy_features(state(h))
    assert f.mean == 2.0
    assert f.maximum == 2.7
    assert f.last == 2.3
    assert f.recent_mean == pytest.approx(1.5)


def test_acceptance_features_include_length_and_all_accepted_rate():
    h = [
        make_step(k=4, accepted=2),
        make_step(k=4, accepted=4),
    ]
    f = acceptance_features(state(h))
    assert f.prev_rate == 1.0
    assert f.recent_rate == pytest.approx(0.75)
    assert f.prev_length == 4.0
    assert f.recent_length == pytest.approx(3.0)
    assert f.all_accepted_rate == pytest.approx(0.5)


def test_feature_shapes_and_combined_has_only_one_bias():
    s = state([make_step()])
    entropy = extract_entropy_features(s)
    acceptance = extract_acceptance_features(s)
    latency = extract_latency_features(s)
    combined = extract_combined_features(s)

    assert entropy.shape == (5,)       # bias + 4
    assert acceptance.shape == (6,)    # bias + 5
    assert latency.shape == (8,)       # bias + 7
    assert combined.shape == (17,)     # one bias + 4 + 5 + 7
    assert combined[0].item() == 1.0


def test_linucb_returns_only_candidate_actions():
    p = LinUCBPolicy(extract_latency_features, k_candidates=(1, 3, 5, 8), seed=0)
    assert p(state([make_step()])) in (1, 3, 5, 8)


def test_freeze_stops_learning_and_exploration():
    p = LinUCBPolicy(
        extract_latency_features,
        k_candidates=(1, 3),
        epsilon=1.0,
        seed=0,
        normalize_features=False,
    )
    st = state()
    for i in range(8):
        st.step_index = i
        k = p(st)
        st.history.append(make_step(step_index=i, k=k, accepted=k, output_tokens=k))
    p.finalize(st)

    A_before = [x.clone() for x in p.A]
    b_before = [x.clone() for x in p.b]
    p.freeze()

    for i in range(8, 12):
        st.step_index = i
        p(st)
        st.history.append(make_step(step_index=i))

    assert p.training is False
    for before, after in zip(A_before, p.A):
        assert torch.equal(before, after)
    for before, after in zip(b_before, p.b):
        assert torch.equal(before, after)


def test_running_normalization_rebuilds_old_samples_in_current_coordinates():
    """As more samples arrive the running mean/std keep changing; A/b for an
    arm must end up reflecting ALL of that arm's samples normalized under the
    CURRENT (final) stats, not a mix of the coordinate systems seen along the
    way when each sample first arrived."""
    p = LinUCBPolicy(
        extract_latency_features, k_candidates=(1,), epsilon=0.0, seed=0,
        reward_fn=lambda step: 1.0,
    )
    st = state()
    raw_features = []
    for draft_s, verify_s in [(0.005, 0.005), (0.020, 0.005), (0.010, 0.030)]:
        k = p(st)
        assert k == 1
        raw_x, _ = p.pending[id(st)]
        raw_features.append(raw_x.clone())
        st.history.append(make_step(k=1, accepted=1, output_tokens=1, draft_s=draft_s, verify_s=verify_s))
    p.finalize(st)  # flushes the third round's observation into A/b

    x_final = [p._normalize(raw) for raw in raw_features]
    dim = len(x_final[0])
    expected_A = torch.eye(dim, dtype=torch.float64) * p.ridge_lambda
    expected_b = torch.zeros(dim, dtype=torch.float64)
    for x in x_final:
        expected_A += torch.outer(x, x)
        expected_b += x
    assert torch.allclose(p.A[0], expected_A, atol=1e-8)
    assert torch.allclose(p.b[0], expected_b, atol=1e-8)

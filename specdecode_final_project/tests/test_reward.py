import pytest

from specdecode.reward import DirectThroughputReward, ThroughputReward, default_reward
from specdecode.speculative import StepResult


def make_step(k=4, accepted=2, output_tokens=3, wall_s=0.02):
    return StepResult(
        step_index=0,
        k_requested=k,
        num_accepted=accepted,
        new_token_ids=[0] * output_tokens,
        draft_time_s=wall_s * 0.6,
        verify_time_s=wall_s * 0.4,
        step_time_s=wall_s,
    )


def test_default_reward_penalizes_rejected_tokens():
    s = make_step(k=5, accepted=2, output_tokens=3, wall_s=0.1)
    assert default_reward(s, rejection_penalty=1.0) == pytest.approx(27.0)


def test_throughput_reward_uses_current_rho_before_update():
    r = ThroughputReward(beta=0.0, ema_tau=0.5, initial_rho=100.0)
    s = make_step(k=2, accepted=2, output_tokens=3, wall_s=0.01)
    assert r(s) == pytest.approx(2.0)


def test_higher_beta_penalizes_rejection_more():
    s = make_step(k=8, accepted=1, output_tokens=2, wall_s=0.02)
    low = ThroughputReward(beta=0.1, initial_rho=0.0)(s)
    high = ThroughputReward(beta=1.0, initial_rho=0.0)(s)
    assert high < low


def test_zero_time_does_not_change_rho():
    r = ThroughputReward(initial_rho=42.0)
    r(make_step(wall_s=0.0))
    assert r.rho_hat == 42.0


def test_reward_instances_do_not_share_state():
    a = ThroughputReward()
    b = ThroughputReward()
    a(make_step(k=1, accepted=1, output_tokens=2, wall_s=0.001))
    assert a.rho_hat != b.rho_hat


def test_direct_throughput_reward_matches_step_tps_minus_rejection_penalty():
    s = make_step(k=5, accepted=2, output_tokens=3, wall_s=0.1)
    r = DirectThroughputReward(beta=0.05)
    assert r(s) == pytest.approx(30.0 - 0.15)

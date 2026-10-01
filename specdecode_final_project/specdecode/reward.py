from __future__ import annotations

from .speculative import StepResult


def default_reward(step: StepResult, rejection_penalty: float = 0.25) -> float:
    """Direct step throughput with a rejection penalty."""
    return step.tokens_per_second - rejection_penalty * step.num_rejected


class DirectThroughputReward:
    """Legacy unweighted step-rate reward, retained for comparisons only."""

    def __init__(self, beta: float = 0.10):
        self.beta = beta

    def __call__(self, step: StepResult) -> float:
        return step.tokens_per_second - self.beta * step.num_rejected


class ThroughputRateReward:
    """Duration-weighted regression of useful tokens per second.

    With an intercept-only fit (without ridge), the predicted value is
    sum(tokens - beta * rejected) / sum(seconds), not mean(tokens / seconds).
    LinUCBPolicy consumes sample_weight as well as __call__.
    This is a local service-rate proxy, not a guarantee of system throughput.
    Beta is measured in useful-token equivalents per rejected token; old beta
    validation results for DirectThroughputReward do not transfer to this scale.
    """

    def __init__(self, beta: float = 0.0):
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        self.beta = beta

    def sample_weight(self, step: StepResult) -> float:
        return max(step.step_time_s, 0.0)

    def __call__(self, step: StepResult) -> float:
        duration = self.sample_weight(step)
        if duration == 0:
            return 0.0
        return (len(step.new_token_ids) - self.beta * step.num_rejected) / duration


class ThroughputReward:
    """Original online reward kept for reproducibility and comparison."""

    def __init__(self, beta: float = 0.25, ema_tau: float = 0.15, initial_rho: float = 0.0):
        self.beta = beta
        self.ema_tau = ema_tau
        self.rho_hat = initial_rho

    def __call__(self, step: StepResult) -> float:
        num_tokens = len(step.new_token_ids)
        reward = num_tokens - self.rho_hat * step.step_time_s - self.beta * step.num_rejected

        if step.step_time_s > 0:
            self.rho_hat += self.ema_tau * (step.tokens_per_second - self.rho_hat)
        return reward

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch

from .reward import ThroughputRateReward
from .speculative import GenerationState, StepResult

WINDOW = 5


def _recent(values: Sequence[float], window: int = WINDOW):
    if not values:
        return 0.0, 0.0, 0.0
    prev = float(values[-1])
    part = values[-window:]
    avg = float(sum(part) / len(part))
    return prev, avg, prev - avg


@dataclass
class EntropyFeatures:
    mean: float
    maximum: float
    last: float
    recent_mean: float

    def values(self):
        return [self.mean, self.maximum, self.last, self.recent_mean]


def entropy_features(state: GenerationState) -> EntropyFeatures:
    if not state.history:
        return EntropyFeatures(0.0, 0.0, 0.0, 0.0)
    s = state.history[-1]
    recent = [x.draft_entropy_mean for x in state.history[-WINDOW:]]
    return EntropyFeatures(
        s.draft_entropy_mean,
        s.draft_entropy_max,
        s.draft_entropy_last,
        sum(recent) / len(recent),
    )


@dataclass
class AcceptanceFeatures:
    prev_rate: float
    recent_rate: float
    prev_length: float
    recent_length: float
    all_accepted_rate: float

    def values(self):
        return [self.prev_rate, self.recent_rate, self.prev_length, self.recent_length, self.all_accepted_rate]


def acceptance_features(state: GenerationState) -> AcceptanceFeatures:
    if not state.history:
        return AcceptanceFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

    recent = state.history[-WINDOW:]
    last = recent[-1]
    rates = [s.acceptance_rate for s in recent]
    lengths = [float(s.num_accepted) for s in recent]
    all_rate = sum(float(s.all_accepted) for s in recent) / len(recent)
    return AcceptanceFeatures(
        last.acceptance_rate,
        sum(rates) / len(rates),
        float(last.num_accepted),
        sum(lengths) / len(lengths),
        all_rate,
    )


@dataclass
class LatencyFeatures:
    prev_k: float
    step_ms: float
    recent_step_ms: float
    draft_ms_per_token: float
    verify_ms_per_token: float
    draft_verify_ratio: float
    output_ms_per_token: float

    def values(self):
        return [
            self.prev_k,
            self.step_ms,
            self.recent_step_ms,
            self.draft_ms_per_token,
            self.verify_ms_per_token,
            self.draft_verify_ratio,
            self.output_ms_per_token,
        ]


def latency_features(state: GenerationState) -> LatencyFeatures:
    if not state.history:
        return LatencyFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    last = state.history[-1]
    recent = state.history[-WINDOW:]
    # For the latency policy we use only measured model compute time.
    # On CUDA, draft_time_s and verify_time_s come from CUDA Events.
    # step_time_s also contains Python/policy/rejection-sampling overhead,
    # so it must not be used as the latency signal.
    draft_ms = last.draft_time_s * 1000.0
    verify_ms = last.verify_time_s * 1000.0
    compute_ms = draft_ms + verify_ms
    recent_compute_ms = sum(
        (s.draft_time_s + s.verify_time_s) * 1000.0 for s in recent
    ) / len(recent)

    k = max(last.k_requested, 1)
    out_tokens = max(len(last.new_token_ids), 1)

    return LatencyFeatures(
        float(last.k_requested),
        compute_ms,
        recent_compute_ms,
        draft_ms / k,
        verify_ms / k,
        draft_ms / max(verify_ms, 1e-6),
        compute_ms / out_tokens,
    )


def _tensor(values: Sequence[float]) -> torch.Tensor:
    # One bias term only. Combined policy does not duplicate it.
    return torch.tensor([1.0, *values], dtype=torch.float64)


def extract_entropy_features(state: GenerationState) -> torch.Tensor:
    return _tensor(entropy_features(state).values())


def extract_acceptance_features(state: GenerationState) -> torch.Tensor:
    return _tensor(acceptance_features(state).values())


def extract_latency_features(state: GenerationState) -> torch.Tensor:
    return _tensor(latency_features(state).values())


def extract_combined_features(state: GenerationState) -> torch.Tensor:
    values = (
        entropy_features(state).values()
        + acceptance_features(state).values()
        + latency_features(state).values()
    )
    return _tensor(values)


class LinUCBPolicy:
    """Small contextual bandit that chooses the next k."""

    def __init__(
        self,
        feature_fn: Callable[[GenerationState], torch.Tensor],
        k_candidates: Sequence[int] = (1, 3, 5, 8),
        alpha: float = 0.8,
        ridge_lambda: float = 1.0,
        epsilon: float = 0.0,
        reward_fn: Optional[Callable[[StepResult], float]] = None,
        training: bool = True,
        seed: Optional[int] = None,
        normalize_features: bool = True,
    ):
        self.feature_fn = feature_fn
        self.k_candidates = list(k_candidates)
        if not self.k_candidates or len(set(self.k_candidates)) != len(self.k_candidates):
            raise ValueError("k_candidates must be nonempty and unique")
        if any(not isinstance(k, int) or k < 1 for k in self.k_candidates):
            raise ValueError("k_candidates must be positive integers")
        if ridge_lambda <= 0 or alpha < 0 or not 0 <= epsilon <= 1:
            raise ValueError("Require ridge_lambda > 0, alpha >= 0 and 0 <= epsilon <= 1")
        self.alpha = alpha
        self.ridge_lambda = ridge_lambda
        self.epsilon = epsilon
        self.reward_fn = reward_fn if reward_fn is not None else ThroughputRateReward()
        self.training = training
        self.normalize_features = normalize_features
        self.rng = random.Random(seed)
        self.lock = threading.Lock()

        self.A: Optional[List[torch.Tensor]] = None
        self.b: Optional[List[torch.Tensor]] = None
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.count = 0
        self.pending = {}
        self.num_observations = 0
        self.action_counts = [0 for _ in self.k_candidates]
        self._theta = None

    def _init(self, dim: int):
        eye = torch.eye(dim, dtype=torch.float64) * self.ridge_lambda
        self.A = [eye.clone() for _ in self.k_candidates]
        self.b = [torch.zeros(dim, dtype=torch.float64) for _ in self.k_candidates]
        # Centered sufficient statistics avoid both replaying the full history
        # and cancellation from subtracting large raw second moments.
        self._weights = [0.0 for _ in self.k_candidates]
        self._centers = [torch.zeros(dim, dtype=torch.float64) for _ in self.k_candidates]
        self._scatter = [torch.zeros((dim, dim), dtype=torch.float64) for _ in self.k_candidates]
        self._reward_means = [0.0 for _ in self.k_candidates]
        self._cross = [torch.zeros(dim, dtype=torch.float64) for _ in self.k_candidates]

    def _update_normalization_stats(self, x: torch.Tensor):
        if not self.normalize_features:
            return
        if self.mean is None:
            self.mean = x.clone()
            self.m2 = torch.zeros_like(x)
            self.count = 1
            return
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_features:
            return x.clone()
        if self.mean is None or self.count < 2:
            out = x.clone()
            out[1:] = 0.0
            return out
        var = self.m2 / max(self.count - 1, 1)
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        out = (x - self.mean) / std
        out[0] = 1.0
        return out

    def _rebuild_model(self):
        """Transform sufficient statistics; cost does not grow with sample count."""
        if self.A is None:
            return
        dim = self.A[0].shape[0]
        scale = torch.ones(dim, dtype=torch.float64)
        if self.normalize_features:
            if self.count < 2:
                scale[1:] = 0.0
            else:
                scale = torch.rsqrt(torch.clamp(self.m2 / (self.count - 1), min=1e-8))
                scale[0] = 1.0
        ridge = torch.eye(dim, dtype=torch.float64) * self.ridge_lambda
        for i, weight in enumerate(self._weights):
            center = self._normalize(self._centers[i])
            covariance = self._scatter[i] * torch.outer(scale, scale)
            self.A[i] = ridge + covariance + weight * torch.outer(center, center)
            self.A[i] = (self.A[i] + self.A[i].T) * 0.5
            self.b[i] = scale * self._cross[i] + weight * center * self._reward_means[i]
        self._theta = None

    def _update_previous(self, state: GenerationState):
        key = id(state)
        pending = self.pending.pop(key, None)
        if pending is None or not state.history or not self.training:
            return
        raw_x, action_idx = pending
        step = state.history[-1]
        # A caller may truncate the requested action at the generation budget.
        # Never credit a different executed k to the selected arm.
        if step.k_requested != self.k_candidates[action_idx]:
            return
        reward = float(self.reward_fn(step))
        weight_fn = getattr(self.reward_fn, "sample_weight", None)
        weight = float(weight_fn(step)) if callable(weight_fn) else 1.0
        if weight <= 0:
            return
        self._update_normalization_stats(raw_x)
        total = self._weights[action_idx] + weight
        delta_x = raw_x - self._centers[action_idx]
        delta_r = reward - self._reward_means[action_idx]
        self._centers[action_idx] += (weight / total) * delta_x
        self._reward_means[action_idx] += (weight / total) * delta_r
        self._scatter[action_idx] += weight * torch.outer(delta_x, raw_x - self._centers[action_idx])
        self._cross[action_idx] += weight * delta_x * (reward - self._reward_means[action_idx])
        self._weights[action_idx] = total
        self.action_counts[action_idx] += 1
        self.num_observations += 1
        self._rebuild_model()

    def __call__(self, state: GenerationState) -> int:
        with self.lock:
            self._update_previous(state)
            raw = self.feature_fn(state)
            if self.A is None:
                self._init(len(raw))
            x = self._normalize(raw)
            eligible = [i for i, k in enumerate(self.k_candidates) if k <= state.remaining_budget]
            if not eligible:
                # No configured arm fits. This final partial step is not trained.
                return min(min(self.k_candidates), state.remaining_budget)
            # Observe each feasible arm at least once, including in concurrent
            # training where some selected actions have not completed yet.
            in_flight = [a for _, a in self.pending.values()]
            unseen = [i for i in eligible if self.action_counts[i] == 0 and i not in in_flight]
            if self.training and unseen:
                action_idx = self.rng.choice(unseen)
            elif self.training and self.rng.random() < self.epsilon:
                action_idx = self.rng.choice(eligible)
            else:
                scores = {}
                for i in eligible:
                    A, b = self.A[i], self.b[i]
                    theta = self._theta[i] if self._theta is not None else torch.linalg.solve(A, b)
                    mean = torch.dot(theta, x)
                    bonus = 0.0
                    if self.training and self.alpha:
                        bonus = self.alpha * torch.sqrt(torch.clamp(x @ torch.linalg.solve(A, x), min=0.0))
                    scores[i] = float(mean + bonus)
                best = max(scores.values())
                ties = [i for i in eligible if scores[i] == best]
                action_idx = self.rng.choice(ties) if self.training else ties[0]

            if self.training:
                self.pending[id(state)] = (raw.clone(), action_idx)
            result = self.k_candidates[action_idx]
        return result

    def finalize(self, state: GenerationState):
        with self.lock:
            self._update_previous(state)
            self.pending.pop(id(state), None)

    def observe(self, state: GenerationState):
        """Learn after the measured round, outside its reward timing window."""
        with self.lock:
            self._update_previous(state)

    def freeze(self):
        """Stop learning/exploration and only use the learned policy."""
        with self.lock:
            if self.pending:
                raise RuntimeError("Finish outstanding requests before freezing the policy")
            self.training = False
            if self.A is not None:
                self._theta = [torch.linalg.solve(A, b) for A, b in zip(self.A, self.b)]

    def unfreeze(self):
        with self.lock:
            self.training = True
            self._theta = None


def make_entropy_only_policy(**kwargs):
    return LinUCBPolicy(extract_entropy_features, **kwargs)


def make_acceptance_history_policy(**kwargs):
    return LinUCBPolicy(extract_acceptance_features, **kwargs)


def make_latency_aware_policy(**kwargs):
    return LinUCBPolicy(extract_latency_features, **kwargs)


def make_combined_policy(**kwargs):
    return LinUCBPolicy(extract_combined_features, **kwargs)

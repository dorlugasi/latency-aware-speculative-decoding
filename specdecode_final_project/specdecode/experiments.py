"""Shared training, evaluation guards, and provenance for experiment scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

from .harness import RuntimeRegime, run_under_regime
from .policy import LinUCBPolicy


def train_mixed_policy(decoder, policy, prompts, train_users, tokens, seed):
    if tokens <= 0:
        raise ValueError("Adaptive policies require positive training tokens")
    if not train_users or any(n <= 0 for n in train_users):
        raise ValueError("Training loads must be positive")
    loads = list(train_users)
    random.Random(seed).shuffle(loads)
    for users in loads:
        # A seed changes exploration and load order, not the training dataset.
        batch = [prompts[(users + i) % len(prompts)] for i in range(users)]
        run_under_regime(
            decoder, batch, max_new_tokens=tokens, k=policy,
            regime=RuntimeRegime(f"train_u{users}", users, 0),
            do_sample=False, record_runtime_stats=False,
        )
    policy.freeze()


def require_frozen(policy):
    if isinstance(policy, LinUCBPolicy) and policy.training:
        raise ValueError("Measured evaluation requires a frozen policy")


def policy_snapshot(policy):
    if not isinstance(policy, LinUCBPolicy):
        return {"fixed_k": int(policy)}
    require_frozen(policy)
    return {
        "k_candidates": policy.k_candidates,
        "feature_fn": policy.feature_fn.__name__,
        "alpha": policy.alpha, "epsilon": policy.epsilon,
        "ridge_lambda": policy.ridge_lambda,
        "normalize_features": policy.normalize_features,
        "normalization_count": policy.count,
        "mean": None if policy.mean is None else policy.mean.tolist(),
        "m2": None if policy.m2 is None else policy.m2.tolist(),
        "A": None if policy.A is None else [a.tolist() for a in policy.A],
        "b": None if policy.b is None else [b.tolist() for b in policy.b],
        "reward": type(policy.reward_fn).__name__,
        "beta": getattr(policy.reward_fn, "beta", None),
        "action_counts": policy.action_counts,
        "observations": policy.num_observations,
    }


def policy_id(policy):
    payload = json.dumps(policy_snapshot(policy), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def save_policy_snapshot(policy, output, label):
    directory = Path(str(output) + ".policies")
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = policy_snapshot(policy)
    snapshot["policy_id"] = policy_id(policy)
    (directory / f"{label}.json").write_text(json.dumps(snapshot, indent=2) + "\n")


def write_run_metadata(output, args):
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted([*root.joinpath("specdecode").glob("*.py"), *root.joinpath("scripts").glob("*.py")]):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    versions = {}
    for package in ("torch", "transformers"):
        versions[package] = importlib.metadata.version(package)
    metadata = {
        "protocol": "duration_weighted_rate_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "arguments": vars(args),
        "evaluation_prompt_offset": 0,
        "evaluation_exploration": False,
        "training_load_shape": "one concurrent request per user; finishes drain",
    }
    Path(str(output) + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

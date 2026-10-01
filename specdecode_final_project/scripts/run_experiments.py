from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from specdecode.harness import RuntimeRegime, run_under_regime
from specdecode.models import load_model_and_tokenizer
from specdecode.policy import (
    LinUCBPolicy,
    make_acceptance_history_policy,
    make_combined_policy,
    make_entropy_only_policy,
    make_latency_aware_policy,
)
from specdecode.speculative import SpeculativeDecoder
from specdecode.reward import ThroughputRateReward
from specdecode.experiments import train_mixed_policy, require_frozen, policy_id, save_policy_snapshot, write_run_metadata

FIXED_K = (1, 3, 5, 8)
POLICY_FACTORIES = {
    "entropy": make_entropy_only_policy,
    "acceptance": make_acceptance_history_policy,
    "latency": make_latency_aware_policy,
    "combined": make_combined_policy,
}
VARIANTS = [f"fixed_{k}" for k in FIXED_K] + list(POLICY_FACTORIES)

TRAIN_PROMPTS = [
    "Artificial intelligence systems are increasingly used to",
    "A programmer debugging a difficult problem should",
    "The development of modern processors has enabled",
    "Researchers studying language models found that",
    "In the early days of the internet,",
    "A reliable distributed system must handle",
    "The spacecraft transmitted new data about",
    "One important challenge in machine learning is",
    "The city began changing rapidly after",
    "When learning a complicated technical subject,",
    "The engineer examined the unexpected failure and",
    "A recent scientific experiment demonstrated that",
]

TEST_PROMPTS = [
    "Advances in computer hardware have made it possible to",
    "A large online platform must respond quickly when",
    "The professor explained the difficult concept by",
    "During the mission the control system detected",
    "The software team improved the application after",
    "A new generation of language models may",
    "When many requests arrive at the same time,",
    "The experiment was repeated under different conditions because",
    "Computer scientists often evaluate a system by measuring",
    "The small research laboratory developed a method to",
    "After the unexpected change in workload the system",
    "An efficient inference engine should be able to",
]


@dataclass
class TrialRow:
    users: int
    variant: str
    repetition: int
    wall_time_s: float
    total_output_tokens: int
    total_drafted_tokens: int
    total_accepted_tokens: int
    acceptance_rate: float
    tokens_per_second: float
    avg_k: float
    avg_compute_ms: float
    avg_round_wall_ms: float
    avg_policy_us: float
    policy_id: str


def auto_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_variant(name, seed, reward_beta=0.0, alpha=0.8, epsilon=0.0, ridge_lambda=1.0):
    if name.startswith("fixed_"):
        return int(name.split("_")[1])
    return POLICY_FACTORIES[name](
        seed=seed,
        normalize_features=True,
        reward_fn=ThroughputRateReward(beta=reward_beta),
        alpha=alpha, epsilon=epsilon, ridge_lambda=ridge_lambda,
    )


def prompts_for_users(pool, users, offset=0):
    return [pool[(offset + i) % len(pool)] for i in range(users)]


def warmup(decoder):
    decoder.generate(TRAIN_PROMPTS[0], max_new_tokens=8, k=3, do_sample=False)


def train_policy(decoder, policy, train_users, tokens_per_user, seed):
    """Train on mixed user loads, then freeze before measured evaluation."""
    train_mixed_policy(decoder, policy, TRAIN_PROMPTS, train_users, tokens_per_user, seed)


def write_step_rows(writer, results, users, variant, repetition):
    for user_id, result in enumerate(results):
        for s in result.steps:
            writer.writerow({
                "users": users,
                "variant": variant,
                "repetition": repetition,
                "user_id": user_id,
                "step": s.step_index,
                "k": s.k_requested,
                "accepted": s.num_accepted,
                "rejected": s.num_rejected,
                "acceptance_pct": 100.0 * s.acceptance_rate,
                "output_tokens": len(s.new_token_ids),
                "draft_ms": 1000.0 * s.draft_time_s,
                "verify_ms": 1000.0 * s.verify_time_s,
                "compute_ms": 1000.0 * (s.draft_time_s + s.verify_time_s),
                "round_wall_ms": 1000.0 * s.step_time_s,
                "policy_us": 1e6 * s.policy_time_s,
                "entropy_mean": s.draft_entropy_mean,
                "entropy_max": s.draft_entropy_max,
                "entropy_last": s.draft_entropy_last,
                "draft_ms_per_token": s.draft_ms_per_token,
                "verify_ms_per_token": s.verify_ms_per_token,
                "draft_verify_ratio": s.draft_verify_ratio,
            })


def evaluate_trial(decoder, users, variant, repetition, max_new_tokens, step_writer, k):
    require_frozen(k)
    regime = RuntimeRegime(name=f"users_{users}", num_streams=users, contention_intensity=0)
    prompts = prompts_for_users(TEST_PROMPTS, users)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    results = run_under_regime(
        decoder, prompts, max_new_tokens=max_new_tokens, k=k, regime=regime,
        do_sample=False, record_runtime_stats=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - start

    write_step_rows(step_writer, results, users, variant, repetition)
    steps = [s for r in results for s in r.steps]
    drafted = sum(s.k_requested for s in steps)
    accepted = sum(s.num_accepted for s in steps)
    output_tokens = sum(len(r.token_ids) for r in results)
    return TrialRow(
        users=users, variant=variant, repetition=repetition, wall_time_s=wall,
        total_output_tokens=output_tokens, total_drafted_tokens=drafted,
        total_accepted_tokens=accepted,
        acceptance_rate=accepted / drafted if drafted else 0.0,
        tokens_per_second=output_tokens / wall if wall else 0.0,
        avg_k=statistics.mean(s.k_requested for s in steps),
        avg_compute_ms=statistics.mean((s.draft_time_s + s.verify_time_s) * 1000 for s in steps),
        avg_round_wall_ms=statistics.mean(s.step_time_s * 1000 for s in steps),
        avg_policy_us=statistics.mean(s.policy_time_s * 1e6 for s in steps),
        policy_id=policy_id(k),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--target", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--device", default=None)
    p.add_argument("--users", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--train-tokens", type=int, default=80)
    p.add_argument("--train-users", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    p.add_argument("--out", default="experiment_results.csv")
    p.add_argument("--steps-out", default="experiment_steps.csv")
    p.add_argument("--reward-beta", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    args = p.parse_args()
    if args.repetitions <= 0 or args.max_new_tokens <= 0 or any(n <= 0 for n in args.users + args.train_users):
        p.error("Repetitions, tokens, and user loads must be positive")
    if any(v in POLICY_FACTORIES for v in args.variants) and args.train_tokens <= 0:
        p.error("Adaptive evaluation requires --train-tokens > 0")
    write_run_metadata(args.out, args)

    device = args.device or auto_device()
    print(f"device={device} draft={args.draft} target={args.target}")
    print(f"users={args.users} variants={args.variants}")

    draft_model, draft_tok = load_model_and_tokenizer(args.draft, device=device)
    target_model, target_tok = load_model_and_tokenizer(args.target, device=device)
    if draft_tok.get_vocab() != target_tok.get_vocab():
        raise RuntimeError("Draft and target tokenizers differ.")

    decoder = SpeculativeDecoder(draft_model, target_model, target_tok, device=device)
    warmup(decoder)

    step_fields = [
        "users", "variant", "repetition", "user_id", "step", "k",
        "accepted", "rejected", "acceptance_pct", "output_tokens",
        "draft_ms", "verify_ms", "compute_ms", "round_wall_ms", "policy_us",
        "entropy_mean", "entropy_max", "entropy_last",
        "draft_ms_per_token", "verify_ms_per_token", "draft_verify_ratio",
    ]
    trial_fields = list(TrialRow.__dataclass_fields__)

    with open(args.out, "w", newline="") as tf, open(args.steps_out, "w", newline="") as sf:
        tw = csv.DictWriter(tf, fieldnames=trial_fields)
        sw = csv.DictWriter(sf, fieldnames=step_fields)
        tw.writeheader()
        sw.writeheader()

        # Train each adaptive policy once per repetition, then reuse that exact
        # frozen policy across all evaluation loads. This isolates runtime
        # adaptation from differences caused by retraining a new policy per load.
        total = len(args.variants) * args.repetitions * len(args.users)
        i = 0
        for variant in args.variants:
            for rep in range(args.repetitions):
                k = make_variant(variant, rep, reward_beta=args.reward_beta, alpha=args.alpha,
                                 epsilon=args.epsilon, ridge_lambda=args.ridge_lambda)
                if isinstance(k, LinUCBPolicy):
                    train_policy(decoder, k, args.train_users, args.train_tokens, seed=rep)
                save_policy_snapshot(k, args.out, f"{variant}_seed{rep}")

                eval_users = list(args.users)
                random.Random(1234 + rep).shuffle(eval_users)
                for users in eval_users:
                    i += 1
                    row = evaluate_trial(
                        decoder, users, variant, rep, args.max_new_tokens, sw, k
                    )
                    tw.writerow(asdict(row))
                    tf.flush(); sf.flush()
                    print(
                        f"[{i}/{total}] users={users:<2} {variant:<10} "
                        f"tok/s={row.tokens_per_second:7.2f} "
                        f"accepted={row.total_accepted_tokens}/{row.total_drafted_tokens} "
                        f"({100*row.acceptance_rate:5.1f}%) avg_k={row.avg_k:.2f}",
                        flush=True,
                    )

    print(f"\nwrote {args.out}")
    print(f"wrote {args.steps_out}")


if __name__ == "__main__":
    main()

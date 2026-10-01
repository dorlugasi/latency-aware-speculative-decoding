from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_static(trials, steps, out):
    out.mkdir(parents=True, exist_ok=True)

    summary = trials.groupby(["users", "variant"], as_index=False).agg(
        n=("tokens_per_second", "size"),
        tps_mean=("tokens_per_second", "mean"),
        tps_std=("tokens_per_second", "std"),
        acceptance_pct=("acceptance_rate", lambda x: 100*x.mean()),
        accepted_tokens=("total_accepted_tokens", "mean"),
        drafted_tokens=("total_drafted_tokens", "mean"),
        avg_k=("avg_k", "mean"),
        compute_ms=("avg_compute_ms", "mean"),
        policy_us=("avg_policy_us", "mean"),
    )
    summary.to_csv(out / "static_summary.csv", index=False)

    # Hindsight best fixed-k baseline for each user count.
    fixed = summary[summary["variant"].str.startswith("fixed_")].copy()
    best_fixed = fixed.loc[fixed.groupby("users")["tps_mean"].idxmax(),
                       ["users", "variant", "tps_mean"]].rename(
        columns={"variant": "best_fixed_k", "tps_mean": "best_fixed_tps"}
    )
    summary = summary.merge(best_fixed, on="users", how="left")
    summary["pct_of_best_fixed"] = 100 * summary["tps_mean"] / summary["best_fixed_tps"]
    summary.to_csv(out / "static_summary_with_best_fixed.csv", index=False)

    for metric, ylabel in [
        ("tps_mean", "Tokens / second"),
        ("acceptance_pct", "Accepted draft tokens (%)"),
        ("avg_k", "Average chosen k"),
        ("pct_of_best_fixed", "% of hindsight best fixed-k baseline"),
        ("policy_us", "Policy decision overhead (us)"),
    ]:
        piv = summary.pivot(index="users", columns="variant", values=metric)
        ax = piv.plot(marker="o", figsize=(9, 5))
        ax.set_xlabel("Concurrent users")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs concurrent users")
        plt.tight_layout()
        plt.savefig(out / f"{metric}_vs_users.png", dpi=160)
        plt.close()

    # Detailed accepted-token distribution per k.
    accepted = steps.groupby(["variant", "k", "accepted"], as_index=False).size()
    totals = accepted.groupby(["variant", "k"])["size"].transform("sum")
    accepted["percent_of_steps"] = 100 * accepted["size"] / totals
    accepted.to_csv(out / "accepted_tokens_distribution.csv", index=False)

    print("\nStatic summary:")
    print(summary.to_string(index=False))


def save_dynamic(dynamic, out):
    if dynamic is None:
        return
    out.mkdir(parents=True, exist_ok=True)

    dynamic.to_csv(out / "dynamic_full.csv", index=False)
    for variant, g in dynamic.groupby("variant"):
        g = g.sort_values("phase").copy()
        g["time_index"] = range(len(g))

        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(g["time_index"], g["tokens_per_second"], marker="o", label="throughput")
        ax1.set_xlabel("Dynamic load phase")
        ax1.set_ylabel("Tokens / second")
        ax2 = ax1.twinx()
        ax2.step(g["time_index"], g["users_target"], where="mid", label="users")
        ax2.set_ylabel("Concurrent users")
        ax1.set_title(f"Dynamic non-sinusoidal load: {variant}")
        fig.tight_layout()
        plt.savefig(out / f"dynamic_{variant}.png", dpi=160)
        plt.close()

    # Direct comparison at each load phase, not an artificial stable->unstable ordering.
    pivot = dynamic.pivot_table(
        index=["phase", "users_target"], columns="variant",
        values="tokens_per_second", aggfunc="mean"
    ).reset_index()
    pivot.to_csv(out / "dynamic_policy_comparison.csv", index=False)


def save_oracle_correctness(oracle_df, out):
    if oracle_df is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    correctness = oracle_df.groupby("k", as_index=False).agg(
        exact_match_rate=("exact_match_target", "mean"),
        acceptance_pct=("acceptance_pct", "mean"),
        spec_tps=("spec_tps", "mean"),
        target_tps=("target_tps", "mean"),
    )
    correctness["exact_match_rate"] *= 100
    correctness.to_csv(out / "correctness_summary.csv", index=False)

    best = oracle_df.loc[oracle_df.groupby("prompt_id")["spec_tps"].idxmax(),
                         ["prompt_id", "k", "spec_tps"]]
    best.to_csv(out / "oracle_best_k_per_prompt.csv", index=False)

    print("\nCorrectness / best-fixed-k summary:")
    print(correctness.to_string(index=False))
    print("\nHindsight best fixed k per prompt:")
    print(best.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", default="experiment_results.csv")
    p.add_argument("--steps", default="experiment_steps.csv")
    p.add_argument("--dynamic", default="dynamic_results.csv")
    p.add_argument("--oracle", default="oracle_correctness.csv")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    trials = pd.read_csv(args.trials)
    steps = pd.read_csv(args.steps)
    dynamic = pd.read_csv(args.dynamic) if Path(args.dynamic).exists() else None
    oracle = pd.read_csv(args.oracle) if Path(args.oracle).exists() else None

    out = Path(args.out_dir)
    save_static(trials, steps, out)
    save_dynamic(dynamic, out)
    save_oracle_correctness(oracle, out)

    print(f"\nwrote analysis to {out}/")


if __name__ == "__main__":
    main()

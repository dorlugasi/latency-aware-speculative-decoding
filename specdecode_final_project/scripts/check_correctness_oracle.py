from __future__ import annotations

import argparse
import csv
import time

import torch

from specdecode.models import load_model_and_tokenizer
from specdecode.speculative import SpeculativeDecoder

KS = (1, 3, 5, 8)

PROMPTS = [
    "The future of artificial intelligence is",
    "In a small village by the sea,",
    "The history of computing began with",
    "Climate change is affecting the world by",
    "Scientists recently discovered that",
    "Once the spacecraft reached orbit,",
    "Modern computer systems are designed to",
    "The experiment showed an unexpected result because",
]


@torch.no_grad()
def target_greedy(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    generated = out[0, ids.shape[1]:].tolist()
    return generated, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--target", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--out", default="oracle_correctness.csv")
    args = p.parse_args()

    draft_model, draft_tok = load_model_and_tokenizer(args.draft, device=args.device)
    target_model, target_tok = load_model_and_tokenizer(args.target, device=args.device)
    if draft_tok.get_vocab() != target_tok.get_vocab():
        raise RuntimeError("Draft and target tokenizers differ.")

    decoder = SpeculativeDecoder(draft_model, target_model, target_tok, device=args.device)
    decoder.generate(PROMPTS[0], max_new_tokens=8, k=3, do_sample=False)

    fields = [
        "prompt_id", "k", "exact_match_target", "matched_tokens", "target_tokens",
        "accepted", "drafted", "acceptance_pct", "spec_tps", "target_tps"
    ]
    rows = []

    for prompt_id, prompt in enumerate(PROMPTS):
        target_ids, target_wall = target_greedy(
            target_model, target_tok, prompt, args.max_new_tokens, args.device
        )
        target_tps = len(target_ids) / target_wall

        best_k = None
        best_tps = -1.0
        for k in KS:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = decoder.generate(
                prompt, max_new_tokens=args.max_new_tokens, k=k,
                do_sample=False, record_runtime_stats=False
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            matched = sum(a == b for a, b in zip(result.token_ids, target_ids))
            exact = result.token_ids == target_ids
            accepted = sum(s.num_accepted for s in result.steps)
            drafted = sum(s.k_requested for s in result.steps)
            spec_tps = len(result.token_ids) / wall

            rows.append({
                "prompt_id": prompt_id,
                "k": k,
                "exact_match_target": exact,
                "matched_tokens": matched,
                "target_tokens": len(target_ids),
                "accepted": accepted,
                "drafted": drafted,
                "acceptance_pct": 100.0 * accepted / drafted if drafted else 0.0,
                "spec_tps": spec_tps,
                "target_tps": target_tps,
            })
            if spec_tps > best_tps:
                best_tps = spec_tps
                best_k = k

        print(
            f"prompt={prompt_id} exact_correctness="
            f"{all(r['exact_match_target'] for r in rows if r['prompt_id']==prompt_id)} "
            f"best_fixed_k={best_k} best_fixed_tps={best_tps:.2f}",
            flush=True,
        )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {args.out}")
    print("This is a hindsight best-fixed-k baseline over {1,3,5,8}; it is not a per-step oracle.")
    print("exact_match_target checks that greedy speculative decoding produces the same tokens as target-only greedy decoding.")


if __name__ == "__main__":
    main()

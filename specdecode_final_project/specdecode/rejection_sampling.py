"""Sampling math for speculative decoding's accept/reject rule.

Implements the modified rejection sampling scheme from Leviathan et al.
("Fast Inference from Transformers via Speculative Decoding") / Chen et al.
("Accelerating Large Language Model Decoding with Speculative Sampling"):
a draft token sampled from q is accepted with probability min(1, p(x)/q(x));
on rejection, the replacement is drawn from norm(max(p - q, 0)). This keeps
the marginal output distribution identical to sampling from the target
model p alone.
"""

from __future__ import annotations

import torch

_RESIDUAL_MASS_EPS = 1e-8


def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax with temperature, computed in float32 for numerical stability."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use do_sample=False for greedy decoding")
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_from_probs(probs: torch.Tensor) -> int:
    return int(torch.multinomial(probs, num_samples=1).item())


def residual_probs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(p - q, 0)), falling back to p if the residual mass is ~0.

    The fallback only triggers when p and q nearly coincide everywhere, i.e.
    exactly the regime where rejecting was a rare, low-consequence event.
    """
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if total <= _RESIDUAL_MASS_EPS:
        return p
    return residual / total


def acceptance_prob(p_token: torch.Tensor, q_token: torch.Tensor) -> float:
    return min(1.0, (p_token / q_token).item())


def entropy_nats(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector, in nats."""
    return float(-(probs * torch.log(probs.clamp_min(eps))).sum().item())

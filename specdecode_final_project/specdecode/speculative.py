from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .cache_utils import crop_cache
from .instrumentation import RuntimeSnapshot, save_steps_csv, snapshot_runtime, steps_to_dicts
from .rejection_sampling import acceptance_prob, entropy_nats, logits_to_probs, residual_probs, sample_from_probs


@dataclass
class StepResult:
    """Measurements and output of one speculative round."""

    step_index: int
    k_requested: int
    num_accepted: int
    new_token_ids: List[int]
    draft_time_s: float
    verify_time_s: float
    step_time_s: float
    policy_time_s: float = 0.0
    device_type: str = ""
    gpu_utilization_pct: Optional[float] = None
    gpu_mem_allocated_mb: Optional[float] = None
    gpu_mem_reserved_mb: Optional[float] = None
    draft_entropy_mean: float = 0.0
    draft_entropy_max: float = 0.0
    draft_entropy_last: float = 0.0
    started_at_s: float = 0.0
    decision_at_s: float = 0.0
    ended_at_s: float = 0.0

    @property
    def all_accepted(self) -> bool:
        return self.num_accepted == self.k_requested

    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / self.k_requested if self.k_requested else 0.0

    @property
    def num_rejected(self) -> int:
        return self.k_requested - self.num_accepted

    @property
    def tokens_per_second(self) -> float:
        return len(self.new_token_ids) / self.step_time_s if self.step_time_s > 0 else 0.0

    @property
    def draft_ms_per_token(self) -> float:
        return self.draft_time_s * 1000.0 / max(self.k_requested, 1)

    @property
    def verify_ms_per_token(self) -> float:
        return self.verify_time_s * 1000.0 / max(self.k_requested, 1)

    @property
    def draft_verify_ratio(self) -> float:
        return self.draft_time_s / max(self.verify_time_s, 1e-9)


@dataclass
class GenerationState:
    step_index: int
    generated_token_ids: List[int]
    max_new_tokens: int
    history: List[StepResult] = field(default_factory=list)

    @property
    def remaining_budget(self) -> int:
        return self.max_new_tokens - len(self.generated_token_ids)


@dataclass
class GenerationResult:
    token_ids: List[int]
    text: str
    steps: List[StepResult]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def acceptance_rate(self) -> float:
        drafted = sum(s.k_requested for s in self.steps)
        accepted = sum(s.num_accepted for s in self.steps)
        return accepted / drafted if drafted else 0.0

    @property
    def total_time_s(self) -> float:
        return sum(s.step_time_s for s in self.steps)

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.total_time_s if self.total_time_s > 0 else 0.0

    def to_records(self) -> List[dict]:
        return steps_to_dicts(self.steps)

    def save_csv(self, path: str) -> None:
        save_steps_csv(self.steps, path)


KSelector = Union[int, Callable[[GenerationState], int]]


def _truncate_at_eos(token_ids: List[int], eos_ids: set) -> Tuple[List[int], bool]:
    for idx, tok in enumerate(token_ids):
        if tok in eos_ids:
            return token_ids[: idx + 1], True
    return token_ids, False


def _sync_device(device: str) -> None:
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()


def _timed_call(device: str, fn):
    """Time GPU work with CUDA events; use wall-clock timing elsewhere."""
    if torch.device(device).type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        return result, start.elapsed_time(end) / 1000.0

    _sync_device(device)
    start = time.perf_counter()
    result = fn()
    _sync_device(device)
    return result, time.perf_counter() - start


class SpeculativeDecoder:
    """Draft/target speculative decoding with exact rejection sampling."""

    def __init__(self, draft_model: PreTrainedModel, target_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase, device: str = "cuda"):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        k: KSelector = 4,
        temperature: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[Union[int, Sequence[int]]] = None,
        record_runtime_stats: bool = True,
        on_step: Optional[Callable[[StepResult], None]] = None,
    ) -> GenerationResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_ids = set()
        elif isinstance(eos_token_id, int):
            eos_ids = {eos_token_id}
        else:
            eos_ids = set(eos_token_id)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        committed_len = max(prompt_len - 1, 0)

        if committed_len:
            prefix = prompt_ids[:, :committed_len]
            draft_cache = self.draft_model(input_ids=prefix, use_cache=True).past_key_values
            target_cache = self.target_model(input_ids=prefix, use_cache=True).past_key_values
        else:
            draft_cache = None
            target_cache = None

        pending_token = prompt_ids[:, -1:]
        generated: List[int] = []
        steps: List[StepResult] = []
        state = GenerationState(0, generated, max_new_tokens)

        finished = False
        step_index = 0
        while state.remaining_budget > 0 and not finished:
            state.step_index = step_index
            round_start = time.perf_counter()
            _sync_device(self.device)

            policy_start = time.perf_counter()
            k_this_step = k(state) if callable(k) else int(k)
            policy_time_s = time.perf_counter() - policy_start
            k_this_step = max(1, min(k_this_step, state.remaining_budget))

            draft_token_ids: List[int] = []
            draft_probs: List[torch.Tensor] = []
            entropies: List[float] = []

            def run_draft():
                nonlocal draft_cache
                cur_input = pending_token
                cur_cache = draft_cache
                for _ in range(k_this_step):
                    out = self.draft_model(input_ids=cur_input, past_key_values=cur_cache, use_cache=True)
                    cur_cache = out.past_key_values
                    logits = out.logits[0, -1]
                    probs = logits_to_probs(logits, temperature)
                    entropies.append(entropy_nats(probs))
                    if do_sample:
                        next_id = sample_from_probs(probs)
                        draft_probs.append(probs)
                    else:
                        next_id = int(torch.argmax(logits).item())
                    draft_token_ids.append(next_id)
                    cur_input = torch.tensor([[next_id]], device=self.device)
                draft_cache = cur_cache

            _, draft_time_s = _timed_call(self.device, run_draft)

            draft_tensor = torch.tensor([draft_token_ids], device=self.device)
            verify_input = torch.cat([pending_token, draft_tensor], dim=1)

            def run_verify():
                return self.target_model(input_ids=verify_input, past_key_values=target_cache, use_cache=True)

            target_out, verify_time_s = _timed_call(self.device, run_verify)
            target_cache = target_out.past_key_values
            target_logits = target_out.logits[0]

            num_accepted = 0
            extra_token_id: Optional[int] = None
            for i, draft_id in enumerate(draft_token_ids):
                if do_sample:
                    p_i = logits_to_probs(target_logits[i], temperature)
                    q_i = draft_probs[i]
                    if random.random() < acceptance_prob(p_i[draft_id], q_i[draft_id]):
                        num_accepted += 1
                        continue
                    extra_token_id = sample_from_probs(residual_probs(p_i, q_i))
                    break
                else:
                    target_id = int(torch.argmax(target_logits[i]).item())
                    if target_id == draft_id:
                        num_accepted += 1
                        continue
                    extra_token_id = target_id
                    break
            else:
                bonus_logits = target_logits[k_this_step]
                if do_sample:
                    extra_token_id = sample_from_probs(logits_to_probs(bonus_logits, temperature))
                else:
                    extra_token_id = int(torch.argmax(bonus_logits).item())

            new_committed_len = committed_len + num_accepted + 1
            target_cache = crop_cache(target_cache, new_committed_len)

            if num_accepted == k_this_step:
                def run_catchup():
                    nonlocal draft_cache
                    catchup = torch.tensor([[draft_token_ids[-1]]], device=self.device)
                    out = self.draft_model(input_ids=catchup, past_key_values=draft_cache, use_cache=True)
                    draft_cache = out.past_key_values

                # This extra draft forward pass is real draft-model compute (needed
                # to catch the draft cache up before the next round), so its time
                # must count toward draft_time_s, not vanish into step overhead.
                _, catchup_time_s = _timed_call(self.device, run_catchup)
                draft_time_s += catchup_time_s

            draft_cache = crop_cache(draft_cache, new_committed_len)
            committed_len = new_committed_len
            pending_token = torch.tensor([[extra_token_id]], device=self.device)

            new_token_ids = draft_token_ids[:num_accepted] + [extra_token_id]
            new_token_ids = new_token_ids[:state.remaining_budget]
            new_token_ids, eos_hit = _truncate_at_eos(new_token_ids, eos_ids)
            generated.extend(new_token_ids)

            _sync_device(self.device)
            round_end = time.perf_counter()
            step_time_s = round_end - round_start
            runtime = snapshot_runtime(self.device) if record_runtime_stats else RuntimeSnapshot(torch.device(self.device).type)

            step = StepResult(
                step_index=step_index,
                k_requested=k_this_step,
                num_accepted=num_accepted,
                new_token_ids=list(new_token_ids),
                draft_time_s=draft_time_s,
                verify_time_s=verify_time_s,
                step_time_s=step_time_s,
                policy_time_s=policy_time_s,
                device_type=runtime.device_type,
                gpu_utilization_pct=runtime.gpu_utilization_pct,
                gpu_mem_allocated_mb=runtime.gpu_mem_allocated_mb,
                gpu_mem_reserved_mb=runtime.gpu_mem_reserved_mb,
                draft_entropy_mean=sum(entropies) / len(entropies) if entropies else 0.0,
                draft_entropy_max=max(entropies) if entropies else 0.0,
                draft_entropy_last=entropies[-1] if entropies else 0.0,
                started_at_s=round_start,
                decision_at_s=policy_start,
                ended_at_s=round_end,
            )
            steps.append(step)
            state.history.append(step)
            if on_step is not None:
                on_step(step)
            observe = getattr(k, "observe", None)
            if callable(observe):
                observe(state)
            step_index += 1
            finished = eos_hit

        finalize = getattr(k, "finalize", None)
        if callable(finalize):
            finalize(state)

        return GenerationResult(generated, self.tokenizer.decode(generated, skip_special_tokens=True), steps)

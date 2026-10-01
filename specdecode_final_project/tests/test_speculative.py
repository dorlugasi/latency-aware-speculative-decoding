import torch

import specdecode.speculative as speculative
from _helpers import hf_greedy_reference
from specdecode.speculative import GenerationState, SpeculativeDecoder, _truncate_at_eos

PROMPT = "the quick brown fox"


def test_truncate_at_eos():
    tokens, hit = _truncate_at_eos([10, 20, 30, 40], {30})
    assert tokens == [10, 20, 30]
    assert hit is True


def test_greedy_matches_target_with_mismatched_draft(target_model, noisy_draft_model, tokenizer):
    decoder = SpeculativeDecoder(noisy_draft_model, target_model, tokenizer, device="cpu")
    result = decoder.generate(PROMPT, max_new_tokens=16, k=4, do_sample=False)
    reference = hf_greedy_reference(target_model, tokenizer, PROMPT, 16)
    assert result.token_ids == reference


def test_greedy_matches_target_under_dynamic_k(target_model, noisy_draft_model, tokenizer):
    schedule = [1, 3, 5, 8]

    def selector(state: GenerationState):
        return schedule[state.step_index % len(schedule)]

    decoder = SpeculativeDecoder(noisy_draft_model, target_model, tokenizer, device="cpu")
    result = decoder.generate(PROMPT, max_new_tokens=16, k=selector, do_sample=False)
    reference = hf_greedy_reference(target_model, tokenizer, PROMPT, 16)
    assert result.token_ids == reference
    assert all(1 <= s.k_requested <= 8 for s in result.steps)


def test_max_new_tokens_is_hard_cap(target_model, noisy_draft_model, tokenizer):
    decoder = SpeculativeDecoder(noisy_draft_model, target_model, tokenizer, device="cpu")
    result = decoder.generate(PROMPT, max_new_tokens=7, k=8, do_sample=False, eos_token_id=None)
    assert len(result.token_ids) == 7


def test_entropy_is_recorded_for_entire_draft_round(target_model, noisy_draft_model, tokenizer):
    decoder = SpeculativeDecoder(noisy_draft_model, target_model, tokenizer, device="cpu")
    result = decoder.generate(PROMPT, max_new_tokens=8, k=4, do_sample=False)
    for s in result.steps:
        assert s.draft_entropy_mean >= 0
        assert s.draft_entropy_max >= s.draft_entropy_mean - 1e-6
        assert s.draft_entropy_last >= 0


def test_step_measurements_are_sane(target_model, noisy_draft_model, tokenizer):
    decoder = SpeculativeDecoder(noisy_draft_model, target_model, tokenizer, device="cpu")
    result = decoder.generate(PROMPT, max_new_tokens=8, k=3, do_sample=False)
    for s in result.steps:
        assert s.draft_time_s >= 0
        assert s.verify_time_s >= 0
        assert s.step_time_s >= s.policy_time_s
        assert 0 <= s.num_accepted <= s.k_requested
        assert 0 <= s.acceptance_rate <= 1


def test_draft_time_includes_catchup_forward_pass(target_model, tokenizer, monkeypatch):
    """When every draft token is accepted, the draft model needs an extra
    catch-up forward pass to prep its cache for the next round. That time
    must be added to draft_time_s, not silently dropped into step overhead."""
    real_timed_call = speculative._timed_call
    calls = []

    def spy(device, fn):
        result, elapsed = real_timed_call(device, fn)
        calls.append(elapsed)
        return result, elapsed

    monkeypatch.setattr(speculative, "_timed_call", spy)

    # Using the same model as draft and target guarantees full acceptance
    # every round (draft and target argmax always agree), so the catchup
    # path is exercised on every step.
    decoder = SpeculativeDecoder(target_model, target_model, tokenizer, device="cpu")
    result = decoder.generate(PROMPT, max_new_tokens=6, k=2, do_sample=False)

    num_all_accepted = sum(s.all_accepted for s in result.steps)
    assert num_all_accepted > 0
    # One _timed_call for draft, one for verify, plus one more for the
    # catchup forward pass on every fully-accepted round.
    assert len(calls) == 2 * len(result.steps) + num_all_accepted

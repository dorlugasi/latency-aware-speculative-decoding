import pytest

from _helpers import hf_greedy_reference
from specdecode.harness import RuntimeRegime, run_concurrent_streams, run_under_regime
from specdecode.speculative import SpeculativeDecoder

PROMPTS = ["the quick brown fox", "once upon a time", "hello there my friend"]


def test_concurrent_users_preserve_each_users_greedy_output(target_model, noisy_draft_model, tokenizer):
    decoder = SpeculativeDecoder(noisy_draft_model, target_model, tokenizer, device="cpu")
    results = run_concurrent_streams(decoder, PROMPTS, max_new_tokens=10, k=3, do_sample=False)
    assert len(results) == len(PROMPTS)
    for prompt, result in zip(PROMPTS, results):
        assert result.token_ids == hf_greedy_reference(target_model, tokenizer, prompt, 10)


def test_empty_concurrent_request_returns_empty():
    decoder = SpeculativeDecoder(None, None, None, device="cpu")
    assert run_concurrent_streams(decoder, [], max_new_tokens=10, k=3) == []


def test_mps_multi_user_run_is_rejected_before_model_use():
    decoder = SpeculativeDecoder(None, None, None, device="mps")
    with pytest.raises(RuntimeError, match="MPS"):
        run_concurrent_streams(decoder, ["a", "b"], max_new_tokens=5, k=3)


def test_user_load_regime_without_artificial_gpu_contention(target_model, noisy_draft_model, tokenizer):
    decoder = SpeculativeDecoder(noisy_draft_model, target_model, tokenizer, device="cpu")
    regime = RuntimeRegime(name="users_2", num_streams=2, contention_intensity=0)
    results = run_under_regime(
        decoder, PROMPTS[:2], max_new_tokens=8, k=3, regime=regime, do_sample=False
    )
    assert len(results) == 2

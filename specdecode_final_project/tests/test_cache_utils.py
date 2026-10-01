from specdecode.cache_utils import cache_length, crop_cache


def test_none_cache_has_zero_length():
    assert cache_length(None) == 0


def test_crop_cache_noop_when_requested_length_is_longer(target_model, tokenizer):
    ids = tokenizer("hello world", return_tensors="pt").input_ids
    out = target_model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    before = cache_length(cache)
    cropped = crop_cache(cache, before + 5)
    assert cache_length(cropped) == before

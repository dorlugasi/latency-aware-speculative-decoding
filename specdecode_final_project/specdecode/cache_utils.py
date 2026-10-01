"""Helpers for truncating a Hugging Face KV cache back to a target length.

Needed because the draft/verify forward passes speculatively extend the
cache with tokens that may end up rejected; once we know how many tokens
were actually accepted, both the draft and target caches must be cropped
back before the next round.
"""

from __future__ import annotations

from transformers.cache_utils import Cache


def cache_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, Cache):
        return past_key_values.get_seq_length()
    # legacy tuple-of-tuples format: layer -> (key, value), shape [batch, heads, seq, head_dim]
    return past_key_values[0][0].shape[-2]


def crop_cache(past_key_values, target_length: int):
    """Truncate `past_key_values` to exactly `target_length` tokens.

    A no-op if the cache is already at or below `target_length`.
    """
    if past_key_values is None:
        return None
    current_length = cache_length(past_key_values)
    tokens_to_remove = current_length - target_length
    if tokens_to_remove <= 0:
        return past_key_values
    if isinstance(past_key_values, Cache):
        # Cache.crop takes a negative count to remove that many trailing
        # tokens; a positive value is a deprecated "absolute length" call
        # slated for removal, and silently does the wrong thing here.
        past_key_values.crop(-tokens_to_remove)
        return past_key_values
    return tuple(tuple(t[..., :target_length, :] for t in layer) for layer in past_key_values)

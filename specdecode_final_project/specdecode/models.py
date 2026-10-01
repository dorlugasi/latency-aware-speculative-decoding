"""Model/tokenizer loading for draft and target models."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_model_and_tokenizer(
    name_or_path: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM + tokenizer, in eval mode, on the given device.

    `dtype` defaults to float16 on cuda/mps and float32 on cpu (float16 matmul
    is unsupported/slow on CPU).
    """
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer

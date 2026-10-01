import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="session")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        pytest.skip(f"tiny Hugging Face test model is not available: {e}")


@pytest.fixture(scope="session")
def target_model():
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    except Exception as e:
        pytest.skip(f"tiny Hugging Face test model is not available: {e}")
    model.eval()
    return model


@pytest.fixture(scope="session")
def noisy_draft_model(target_model):
    draft = copy.deepcopy(target_model)
    generator = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for p in draft.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=p.dtype)
            p.add_(noise * 0.5)
    draft.eval()
    return draft

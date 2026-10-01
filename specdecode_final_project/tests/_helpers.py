import torch


def hf_greedy_reference(model, tokenizer, prompt, max_new_tokens):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out[0, ids.shape[1]:].tolist()

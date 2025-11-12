import gc

import torch

from transformer_lens import HookedTransformer
import numpy as np
import random


def set_seed(seed: int = 182):
    """
    Util function to set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    else:
        return torch.device("cpu")


def get_model(model_name: str = "Qwen/Qwen3-1.7B", device: torch.device = get_device()):
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    return model


def clear_model(model: HookedTransformer = None):
    """
    Clear/reset model checkpoint shards from MPS device memory.

    Args:
        model: Optional model object to delete. If None, only clears cache.
    """
    if model is not None:
        del model

    # Force garbage collection to free Python references
    gc.collect()

    # Clear MPS cache (similar to torch.cuda.empty_cache() for CUDA)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_model_parameters(model: HookedTransformer, include: list[str] = None):
    for name, param in model.named_parameters():
        if include is None or any(include_str in name for include_str in include):
            print(name, param.shape)


def generate_response(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 100,
    greedy: bool = True,
):
    """
    Autoregressively generate tokens from a prompt.

    Args:
        model: The HookedTransformer model
        prompt: Input text prompt
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        tokens: Generated tokens tensor with shape [batch_size, seq_len]
    """
    tokens = model.to_tokens(prompt)  # Shape: [batch_size, seq_len]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)  # Shape: [batch_size, seq_len, vocab_size]
            last_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
            probs = torch.softmax(
                last_token_logits, dim=-1
            )  # Shape: [batch_size, vocab_size]
            if greedy:  # Sample the most likely token
                next_token = torch.argmax(
                    probs, dim=-1, keepdim=True
                )  # Shape: [batch_size, 1]
            else:  # Sample a token probabilistically
                next_token = torch.multinomial(
                    probs, num_samples=1
                )  # Shape: [batch_size, 1]
            tokens = torch.cat(
                [tokens, next_token], dim=1
            )  # Shape: [batch_size, seq_len + 1]

    response = model.to_string(tokens)
    return response

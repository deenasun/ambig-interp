import torch

from transformer_lens import HookedTransformer


def set_seed(seed: int = 182):
    """
    Util function to set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)


def get_device():
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    else:
        return torch.device("cpu")


torch.set_grad_enabled(False)
device = get_device()
print("Using device:", device)

MODEL_NAME = "Qwen/Qwen3-1.7B"


def get_model(model_name: str = MODEL_NAME, device: torch.device = device):
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    return model


def print_model_parameters(model: HookedTransformer, include: list[str] = None):
    for name, param in model.named_parameters():
        if include is None or any(include_str in name for include_str in include):
            print(name, param.shape)


tl_model = get_model()
# print_model_parameters(tl_model)

test_text = "Attention is all you need."
test_tokens = tl_model.to_tokens(test_text)
print(test_tokens)
logits, cache = tl_model.run_with_cache(test_tokens)
print(logits)
print(cache)

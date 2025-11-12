from tl_model import get_model, print_model_parameters, set_seed, get_device


set_seed()
device = get_device()
print(f"Device: {device}")

tl_model = get_model(device=device)
print_model_parameters(tl_model)

test_text = "Attention is all you need."
test_tokens = tl_model.to_tokens(test_text)
print(f"Test tokens: {test_tokens}")

logits, cache = tl_model.run_with_cache(test_tokens)
print(f"Logits: {logits}")
print(f"Cache: {cache}")

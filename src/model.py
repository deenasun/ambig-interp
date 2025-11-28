import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import check_memory, clean_up

TOKENIZER_MAX_LENGTH = 256

if torch.accelerator.is_available():
    # Get the current accelerator device (e.g., "cuda:0", "mps:0", "xpu:0")
    device = torch.accelerator.current_accelerator(check_available=True)
    print(f"Using accelerator device: {device}")
else:
    device = torch.device("cpu")
    print("No accelerator found, using CPU.")


def load_model(model_name: str = "meta-llama/Llama-3.2-3B-Instruct", device=torch.accelerator.current_accelerator(check_available=True) if torch.accelerator.is_available() else torch.device("CPU")):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad_token is set and use eos_token if None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model and move to device
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    # Set output_hidden_states = True to extract activations
    model.config.output_hidden_states = True

    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
    chat = [{"role": "user", "content": prompt}]
    # Tokenize with the chat template expected by the model
    tokenizer_outputs = tokenizer.apply_chat_template(chat, return_dict=True, tokenize=True, return_tensors="pt").to(device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "pad_token_id": pad_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**tokenizer_outputs, **generation_kwargs)

    # Calculate number of tokens of input
    input_length = tokenizer_outputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return generated_text


def main():
    tokenizer, model = load_model()
    generated_text = generate(tokenizer, model, "What is the fastest animal?")
    
    print(generated_text)
    
    clean_up()
    check_memory()

if __name__ == "__main__":
    main()

import numpy as np
import torch

TOKENIZER_MAX_LENGTH = 256


def extract_activations(tokenizer, model, prompt: str):
    device = model.device
    model.eval()

    chat = [{"role": "user", "content": prompt}]
    tokenizer_outputs = tokenizer.apply_chat_template(chat, truncation=True, padding=False, max_length=TOKENIZER_MAX_LENGTH, return_dict=True, tokenize=True, return_tensors="pt")
    
    # Extract tensors from dictionary and move to device
    input_ids = tokenizer_outputs["input_ids"].to(device)
    attention_mask = tokenizer_outputs["attention_mask"].to(device)

    # Get last token index
    # Attention_mask = 1 for real tokens. So summing along the sequence dim gives number of real tokens
    last_idx = attention_mask.sum(dim=-1) - 1
    last_idx = last_idx.item()  # Convert to Python int

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states  # (num_layers, batch_size, seq_len, hidden_dim)
        activations = []
        for layer_hidden in hidden_states:
            # Extract last token activation for this layer
            # layer_hidden shape: [batch_size=1, seq_len, hidden_dim]
            last_token_act = layer_hidden[0, last_idx].cpu().to(torch.float16).numpy()  # (1, hidden_dim)
            activations.append(last_token_act)

        # Stack to get (num_layers, hidden_dim)
        return np.stack(activations, axis=0)

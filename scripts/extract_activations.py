import argparse
import json
from pathlib import Path

import numpy as np

from src.activations import extract_activations
from src.model import load_model


def main():
    parser = argparse.ArgumentParser(description="Extract activations")
    parser.add_argument("--input", dest="input_path", type=str, required=True, help="Input JSONL file with questions")
    parser.add_argument("--output", dest="output_path", type=str, required=True, help="Output file to save activations to")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name or path (default: meta-llama/Llama-3.2-3B-Instruct)")
    parser.add_argument("--limit", type=int, default=None, help="Max limit on number of prompts to process")

    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer, model = load_model(model_name=args.model)
    print("Model loaded successfully!")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    activations = {}
    with open(input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if args.limit and len(activations) >= args.limit:
                break

            if not line.strip():
                continue

            try:
                data = json.loads(line)
                line_id = data.get("id")
                if line_id is None:
                    print(f"No id found at line {line_num}")
                    continue

                question = data.get("question")
                if not question:
                    print(f"Empty question at line {line_num}")
                    continue

                label_ambiguous = data.get("label_ambiguous")
                if label_ambiguous is None:
                    print(f"No label found at line {line_num}")
                    continue

                print(f"Processing line {line_num}: {question[:20]}...")
                acts = extract_activations(tokenizer, model, question)  # (num_layers, hidden_dim)
                name = f"{line_id}_label_ambiguous_{label_ambiguous}"
                activations[name] = acts

            except Exception as e:
                import traceback
                print(f"Error loading line {line_num}: {e}")
                print(f"Traceback: {traceback.format_exc()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **activations)
    print(f"\nProcessed {len(activations)} prompts. Activations saved to {output_path}")


if __name__ == "__main__":
    main()

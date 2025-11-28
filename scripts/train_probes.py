import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_project_root

PROJECT_ROOT = get_project_root()
PROBES_DIR = PROJECT_ROOT / "outputs" / "probes"


def train_layerwise_probes(
    X_train,  # shape (N_samples, num_layers, hidden_dim)
    y_train,
    X_val,
    y_val,
):
    probe_performance = []

    def measure_probe_performance(model: LogisticRegression, scaled_X, y):
        y_pred = model.predict(scaled_X)
        accuracy = accuracy_score(y, y_pred)

        y_pred_proba = model.predict_proba(scaled_X)
        # For binary classification, use probabilities of positive class
        roc_auc = roc_auc_score(y, y_pred_proba[:, 1])

        return accuracy, roc_auc

    num_samples, num_layers, hidden_dim = X_train.shape

    for layer in range(num_layers):
        scaler = StandardScaler()
        layer_X_train = scaler.fit_transform(X_train[:, layer, :])
        lr = LogisticRegression(random_state=182, solver="liblinear")
        lr.fit(layer_X_train, y_train)

        layer_X_val = scaler.transform(X_val[:, layer, :])
        train_accuracy, train_roc_auc = measure_probe_performance(lr, layer_X_train, y_train)
        val_accuracy, val_roc_auc = measure_probe_performance(lr, layer_X_val, y_val)

        probe_performance.append(
            {
                "layer": layer,
                "train_accuracy": train_accuracy,
                "train_roc_auc": train_roc_auc,
                "val_accuracy": val_accuracy,
                "val_roc_auc": val_roc_auc,
            }
        )

        coeffs = lr.coef_
        intercept = lr.intercept_

        np.savez_compressed(PROBES_DIR / f"probe_{layer}.npz", coeffs=coeffs, intercept=intercept)

    with open(PROBES_DIR / "probe_performance.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "train_accuracy", "train_roc_auc", "val_accuracy", "val_roc_auc"])
        for perf in probe_performance:
            writer.writerow([perf["layer"], perf["train_accuracy"], perf["train_roc_auc"], perf["val_accuracy"], perf["val_roc_auc"]])

    return probe_performance


def plot_probe_performance(performance_list, output_file):
    """
    Plots accuracies and AUROCs of the probes for each layer
    """
    # Sort by layer number to ensure correct ordering
    sorted_performance = sorted(performance_list, key=lambda x: x["layer"])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    train_accuracies = []
    train_roc_aucs = []
    val_accuracies = []
    val_roc_aucs = []
    for perf in sorted_performance:
        train_accuracies.append(perf["train_accuracy"])
        train_roc_aucs.append(perf["train_roc_auc"])
        val_accuracies.append(perf["val_accuracy"])
        val_roc_aucs.append(perf["val_roc_auc"])
    num_layers = len(sorted_performance)
    xaxis = np.arange(0, num_layers, 1)

    axs[0].plot(xaxis, train_accuracies, marker=".", color="tab:blue", label="Train accuracy")
    axs[0].plot(xaxis, val_accuracies, marker=".", color="tab:orange", label="Val accuracy")
    axs[0].set_title("Layerwise Probe Accuracies")
    axs[0].set_xlabel("Layer")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    axs[1].plot(xaxis, train_roc_aucs, marker=".", color="tab:blue", label="Train AUROC")
    axs[1].plot(xaxis, val_roc_aucs, marker=".", color="tab:orange", label="Val AUROC")
    axs[1].set_title("Layerwise Probe AUROC")
    axs[1].set_xlabel("Layer")
    axs[1].set_ylabel("AUROC")
    axs[1].legend()
    
    fig.suptitle("Layerwise Probe Performance", fontsize=16)

    fig.savefig(output_file)

def main():
    parser = argparse.ArgumentParser(description="Extract activations")
    parser.add_argument("--input", dest="input_path", type=str, required=True, help="Path to activations")
    # parser.add_argument("--output", dest="output_path", type=str, required=True, help="Output file to save activations in")
    # parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name or path")
    # parser.add_argument("--limit", type=int, default=None, help="Max limit on number of prompts to process")
    args = parser.parse_args()

    if not os.path.exists(PROBES_DIR):
        os.makedirs(PROBES_DIR)

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    activations = np.load(input_path)

    y = []
    X = []

    for key, act in activations.items():
        label = key.split("label_ambiguous_")[1]
        X.append(act)
        # Handle both "True"/"False" (from Python bool) and "true"/"false" (from JSON string)
        if label.lower() == "true":
            y.append(1)
        else:
            y.append(0)

    # Convert to numpy arrays
    # X is a list of arrays with shape (num_layers, hidden_dim)
    # Stack them to get shape (N_samples, num_layers, hidden_dim)
    X = np.stack(X, axis=0)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=182, stratify=y)

    print(f"Training layerwise probes on {len(X_train)} training samples and {len(X_val)} validation samples")
    probe_performance = train_layerwise_probes(X_train, y_train, X_val, y_val)

    plot_probe_performance(probe_performance, PROBES_DIR / "layerwise_probes_plots.png")
    print(f"Plot of probe performances saved to {PROBES_DIR / 'layerwise_probes_plots.png'}")
    
    sorted_probes = sorted(probe_performance, key=lambda x: x["val_accuracy"], reverse=True)
    for i, perf in enumerate(sorted_probes):
        print(f"{i}. Layer{perf['layer']}")
        print(f"   Train: Acc={perf['train_accuracy']:.4f}, AUROC={perf['train_roc_auc']:.4f}")
        print(f"   Val:   Acc={perf['val_accuracy']:.4f}, AUROC={perf['val_roc_auc']:.4f}")


if __name__ == "__main__":
    main()

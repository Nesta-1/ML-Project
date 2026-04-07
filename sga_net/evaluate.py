"""
SGA-Net Evaluation Script
---------------------------
Computes metrics for all trained models, generates comparison table,
confusion matrices, and loss curves.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = ["age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"]
MODEL_DISPLAY_NAMES = {
    "vanilla": "Vanilla RoBERTa",
    "concat": "RoBERTa + Concat",
    "sga": "SGA-RoBERTa",
}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    per_class_f1 = f1_score(labels, preds, average=None)

    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, digits=4, output_dict=True
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": {name: float(f) for name, f in zip(LABEL_NAMES, per_class_f1)},
        "classification_report": report,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, model_name, output_path):
    """Save confusion matrix plot."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_loss_curves(histories: dict, output_path: str):
    """Plot training/validation loss curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    for model_key, history in histories.items():
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], "-o", label=f"{name} (train)", markersize=4)
        axes[0].plot(epochs, history["val_loss"], "--s", label=f"{name} (val)", markersize=4)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    for model_key, history in histories.items():
        name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        epochs = range(1, len(history["train_acc"]) + 1)
        axes[1].plot(epochs, history["train_acc"], "-o", label=f"{name} (train)", markersize=4)
        axes[1].plot(epochs, history["val_acc"], "--s", label=f"{name} (val)", markersize=4)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Comparison Table
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_metrics: dict):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"{'MODEL COMPARISON TABLE':^80}")
    print(f"{'='*80}")

    header = f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}"
    for name in LABEL_NAMES:
        short = name[:12]
        header += f" {short:>12}"
    print(header)
    print("-" * len(header))

    for model_key in ["vanilla", "concat", "sga"]:
        if model_key not in all_metrics:
            continue
        m = all_metrics[model_key]
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        row = f"{display_name:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}"
        for name in LABEL_NAMES:
            f1 = m["per_class_f1"].get(name, 0.0)
            row += f" {f1:>12.4f}"
        print(row)

    print(f"{'='*80}")

    # Highlight sarcasm/other class
    print(f"\n{'Key Findings':^80}")
    print("-" * 80)
    if "sga" in all_metrics and "concat" in all_metrics:
        sga_other = all_metrics["sga"]["per_class_f1"].get("other_cyberbullying", 0)
        concat_other = all_metrics["concat"]["per_class_f1"].get("other_cyberbullying", 0)
        diff = sga_other - concat_other
        print(f"  SGA vs Concat on 'other_cyberbullying' F1: {sga_other:.4f} vs {concat_other:.4f} (Δ={diff:+.4f})")

    if "sga" in all_metrics and "vanilla" in all_metrics:
        sga_f1 = all_metrics["sga"]["macro_f1"]
        van_f1 = all_metrics["vanilla"]["macro_f1"]
        diff = sga_f1 - van_f1
        print(f"  SGA vs Vanilla Macro F1: {sga_f1:.4f} vs {van_f1:.4f} (Δ={diff:+.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(checkpoint_dir: str, output_dir: str):
    """Evaluate all available models and generate comparisons."""
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    histories = {}

    for model_key in ["vanilla", "concat", "sga"]:
        model_dir = checkpoint_dir / model_key
        pred_file = model_dir / "test_predictions.npz"
        hist_file = model_dir / "history.json"

        if not pred_file.exists():
            print(f"  ⚠ Skipping {model_key}: no predictions found at {pred_file}")
            continue

        print(f"\n── Evaluating: {MODEL_DISPLAY_NAMES[model_key]} ──")

        # Load predictions
        data = np.load(pred_file)
        preds, labels = data["preds"], data["labels"]

        # Compute metrics
        metrics = compute_metrics(preds, labels)
        all_metrics[model_key] = metrics

        # Print classification report
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Per-class F1:")
        for name, f1 in metrics["per_class_f1"].items():
            print(f"    {name}: {f1:.4f}")

        # Confusion matrix
        plot_confusion_matrix(
            preds, labels,
            MODEL_DISPLAY_NAMES[model_key],
            output_dir / f"confusion_matrix_{model_key}.png"
        )

        # Load training history
        if hist_file.exists():
            with open(hist_file) as f:
                histories[model_key] = json.load(f)

    # Comparison table
    if all_metrics:
        print_comparison_table(all_metrics)

    # Loss curves
    if histories:
        plot_loss_curves(histories, output_dir / "loss_curves.png")

    # Save all metrics
    serializable_metrics = {}
    for k, v in all_metrics.items():
        serializable_metrics[k] = {
            "accuracy": v["accuracy"],
            "macro_f1": v["macro_f1"],
            "per_class_f1": v["per_class_f1"],
        }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    print(f"\n✓ Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGA-Net Evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for evaluation results")
    args = parser.parse_args()

    run_evaluation(args.checkpoint_dir, args.output_dir)

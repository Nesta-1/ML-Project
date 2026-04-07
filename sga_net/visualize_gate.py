"""
SGA-Net Gate Visualization
----------------------------
Visualizes the sentiment gate activations to demonstrate
how sentiment modulates feature representation.

Usage:
    python visualize_gate.py --checkpoint ./checkpoints/sga/best_model.pt
"""

import os
import argparse
import numpy as np
import torch
from transformers import RobertaTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sga_model import SentimentGatedRoberta

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
LABEL_NAMES = ["age", "ethnicity", "gender", "not_cyberbullying", "other_cyberbullying", "religion"]


# ──────────────────────────────────────────────────────────────────────────────
# Core Visualization Function
# ──────────────────────────────────────────────────────────────────────────────

def visualize_gate(
    model: SentimentGatedRoberta,
    text: str,
    tokenizer: RobertaTokenizer = None,
    device: torch.device = None,
    output_path: str = None,
    top_k: int = 50,
):
    """
    Visualize gate activation for a given text input.

    Steps:
        1. Compute VADER sentiment
        2. Forward pass through SGA-RoBERTa
        3. Extract gate vector (768-dim)
        4. Plot heatmap + bar chart of top activations

    Args:
        model: trained SGA-RoBERTa model
        text: input text string
        tokenizer: RoBERTa tokenizer (loaded if None)
        device: torch device
        output_path: path to save visualization
        top_k: number of top gate dimensions to visualize

    Returns:
        gate_values: (768,) numpy array of gate activations
        sentiment: float, compound sentiment score
    """
    if tokenizer is None:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # 1. Compute sentiment
    analyzer = SentimentIntensityAnalyzer()
    compound = analyzer.polarity_scores(text)["compound"]
    sentiment_normalized = (compound + 1.0) / 2.0

    # 2. Tokenize
    enc = tokenizer(
        text, max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    sentiment_tensor = torch.tensor([sentiment_normalized], dtype=torch.float32).to(device)

    # 3. Forward pass
    with torch.no_grad():
        out = model(input_ids, attention_mask, sentiment_score=sentiment_tensor)
    
    gate_values = out["gate_values"].cpu().numpy().squeeze()  # (768,)
    logits = out["logits"].cpu().numpy().squeeze()
    pred_class = LABEL_NAMES[logits.argmax()]

    print(f"\nText: {text[:100]}...")
    print(f"Sentiment (compound): {compound:.4f}")
    print(f"Sentiment (normalized): {sentiment_normalized:.4f}")
    print(f"Predicted class: {pred_class}")
    print(f"Gate stats: mean={gate_values.mean():.4f}, std={gate_values.std():.4f}, "
          f"min={gate_values.min():.4f}, max={gate_values.max():.4f}")

    # 4. Visualize
    if output_path:
        if HAS_PLOTLY:
            _plot_plotly(gate_values, text, compound, pred_class, output_path, top_k)
        else:
            _plot_matplotlib(gate_values, text, compound, pred_class, output_path, top_k)

    return gate_values, compound


def _plot_plotly(gate_values, text, sentiment, pred_class, output_path, top_k):
    """Create interactive Plotly visualization."""
    fig = sp.make_subplots(
        rows=2, cols=1,
        subplot_titles=["Gate Vector Heatmap (768 dims)", f"Top-{top_k} Gate Activations"],
        row_heights=[0.3, 0.7],
        vertical_spacing=0.12,
    )

    # Heatmap — full 768 dims
    fig.add_trace(
        go.Heatmap(
            z=[gate_values],
            colorscale="RdYlBu_r",
            colorbar=dict(title="Activation", x=1.02),
            showscale=True,
        ),
        row=1, col=1,
    )

    # Bar chart — top-k activations
    top_indices = np.argsort(gate_values)[-top_k:][::-1]
    top_values = gate_values[top_indices]
    colors = ["#ef4444" if v > 0.6 else "#3b82f6" if v < 0.4 else "#a3a3a3" for v in top_values]

    fig.add_trace(
        go.Bar(
            x=[f"d{i}" for i in top_indices],
            y=top_values,
            marker_color=colors,
            name="Gate Activation",
        ),
        row=2, col=1,
    )

    short_text = text[:80] + "..." if len(text) > 80 else text
    fig.update_layout(
        title=dict(
            text=f"SGA Gate Visualization<br>"
                 f"<sub>Text: \"{short_text}\" | Sentiment: {sentiment:.3f} | Pred: {pred_class}</sub>",
            font=dict(size=14),
        ),
        height=700,
        width=1200,
        showlegend=False,
        template="plotly_white",
    )

    fig.write_html(output_path)
    print(f"  Saved interactive plot: {output_path}")


def _plot_matplotlib(gate_values, text, sentiment, pred_class, output_path, top_k):
    """Create static matplotlib visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 3]})

    # Heatmap
    im = axes[0].imshow(
        gate_values.reshape(1, -1), aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1
    )
    axes[0].set_title("Gate Vector Heatmap (768 dims)")
    axes[0].set_yticks([])
    plt.colorbar(im, ax=axes[0], orientation="horizontal", pad=0.2, label="Activation")

    # Bar chart
    top_indices = np.argsort(gate_values)[-top_k:][::-1]
    top_values = gate_values[top_indices]
    colors = ["#ef4444" if v > 0.6 else "#3b82f6" if v < 0.4 else "#a3a3a3" for v in top_values]

    axes[1].bar(range(top_k), top_values, color=colors)
    axes[1].set_xlabel("Dimension Index")
    axes[1].set_ylabel("Gate Activation")
    axes[1].set_title(f"Top-{top_k} Gate Activations")
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels([f"d{i}" for i in top_indices], rotation=90, fontsize=6)

    short_text = text[:80] + "..." if len(text) > 80 else text
    fig.suptitle(
        f"SGA Gate — \"{short_text}\"\n"
        f"Sentiment: {sentiment:.3f} | Pred: {pred_class}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Comparative Visualization
# ──────────────────────────────────────────────────────────────────────────────

def compare_sentiments(model, texts: list[str], output_path: str = None):
    """
    Compare gate activations across multiple texts with different sentiments.
    Demonstrates sentiment-dependent feature gating.
    """
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    device = next(model.parameters()).device
    model.eval()

    all_gates = []
    sentiments = []
    analyzer = SentimentIntensityAnalyzer()

    for text in texts:
        gate, sent = visualize_gate(model, text, tokenizer, device)
        all_gates.append(gate)
        sentiments.append(sent)

    if output_path and HAS_PLOTLY:
        fig = go.Figure()
        for i, (text, gate, sent) in enumerate(zip(texts, all_gates, sentiments)):
            short = text[:50] + "..." if len(text) > 50 else text
            fig.add_trace(go.Scatter(
                y=np.sort(gate)[::-1],
                name=f"Sent={sent:.2f}: {short}",
                mode="lines",
                opacity=0.8,
            ))

        fig.update_layout(
            title="Gate Activation Profiles by Sentiment",
            xaxis_title="Dimension Rank",
            yaxis_title="Gate Activation",
            height=500,
            template="plotly_white",
        )
        fig.write_html(output_path)
        print(f"\n  Saved comparison: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGA Gate Visualization")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/sga/best_model.pt",
                        help="Path to SGA model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results/visualizations",
                        help="Output directory")
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentGatedRoberta().to(device)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"No checkpoint found at {args.checkpoint}, using random weights for demo")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Example texts demonstrating sentiment-gate behavior
    test_texts = {
        "positive": "I love how everyone supports each other, such a wonderful community!",
        "negative": "You're so ugly and stupid, nobody likes you, go die",
        "sarcastic": "Oh wow, what a GREAT person you are, bullying kids. So impressive!",
        "neutral": "The school announced new policies regarding student conduct today",
        "implicit": "Some people just shouldn't exist in this world, know what I mean?",
    }

    for label, text in test_texts.items():
        ext = "html" if HAS_PLOTLY else "png"
        out_path = os.path.join(args.output_dir, f"gate_{label}.{ext}")
        visualize_gate(model, text, tokenizer, device, output_path=out_path)

    # Comparison plot
    ext = "html" if HAS_PLOTLY else "png"
    compare_sentiments(
        model,
        list(test_texts.values()),
        output_path=os.path.join(args.output_dir, f"gate_comparison.{ext}"),
    )

    print(f"\n✓ All visualizations saved to {args.output_dir}")

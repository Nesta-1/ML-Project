# SGA-Net: Sentiment-Gated Attention Network

A hybrid framework for robust cyberbullying detection using sentiment-controlled feature gating.

## Novelty Claim

> **Hard-gating mechanism conditions feature relevance on sentiment.** Unlike concatenation-based fusion, SGA-Net uses a learnable gate network that transforms scalar sentiment scores into 768-dimensional binary-like gate vectors. This multiplicative gating selectively amplifies or suppresses RoBERTa feature dimensions based on sentiment context, providing lightweight sentiment guidance that improves robustness on sarcastic/implicit bullying samples.

## Architecture

```
Input Tweet → Clean Text → VADER Sentiment (compound → [0,1])
                                    ↓
                            Gate Network
                            Linear(1→128) → ReLU
                            Linear(128→768) → Sigmoid
                                    ↓
                              gate_vector (B, 768)
                                    ↓
RoBERTa → pooler_output ──×── gate_vector = gated_embedding
                                    ↓
                            Dropout(0.1)
                            Linear(768→6)
                                    ↓
                              logits (6 classes)
```

## Models Compared

| Model | Description |
|-------|-------------|
| **Vanilla RoBERTa** | `RobertaForSequenceClassification` (baseline) |
| **RoBERTa + Concat** | `pooler_output ∥ sentiment` → Linear(769→6) |
| **SGA-RoBERTa** | `pooler_output × gate(sentiment)` → Linear(768→6) |

## Dataset

- **Kaggle Cyberbullying Dataset** (~47.7k tweets)
- 6 balanced classes: `age`, `ethnicity`, `gender`, `not_cyberbullying`, `other_cyberbullying`, `religion`

## Reproducibility

### Setup

```bash
pip install -r requirements.txt
```

### Step 1: Data Pipeline

```bash
python data_pipeline.py --csv ../cyberbullying_tweets.csv --output ./data/processed
```

### Step 2: Train Models

```bash
# Vanilla RoBERTa
python train.py --model vanilla --epochs 3 --batch_size 16

# Concatenation Fusion
python train.py --model concat --epochs 3 --batch_size 16

# SGA-RoBERTa (main model)
python train.py --model sga --epochs 3 --batch_size 16

# Quick smoke test (256 samples, 1 epoch)
python train.py --model sga --epochs 1 --batch_size 16 --max_samples 256
```

### Step 3: Evaluate

```bash
python evaluate.py --checkpoint_dir ./checkpoints --output_dir ./results
```

### Step 4: Visualize Gate

```bash
python visualize_gate.py --checkpoint ./checkpoints/sga/best_model.pt
```

## File Structure

```
sga_net/
├── data_pipeline.py       # Data loading, cleaning, VADER, tokenization
├── baseline_roberta.py    # Vanilla RoBERTa (Baseline 1)
├── concat_model.py        # Concatenation Fusion (Baseline 2)
├── sga_model.py           # SGA-RoBERTa (Main Model)
├── train.py               # Unified training script
├── evaluate.py            # Metrics, confusion matrix, comparison table
├── visualize_gate.py      # Gate activation visualization
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Configuration

| Parameter | Default |
|-----------|---------|
| Seed | 42 |
| Max Length | 128 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Linear warmup (10% steps) |
| Dropout | 0.1 |
| Gate Hidden | 128 |

## Key Design Decisions

1. **Hard gating vs soft attention**: Sigmoid-based gate produces values in [0,1], enabling hard suppression of irrelevant features
2. **VADER as sentiment source**: No additional model overhead; compound score captures sentiment polarity effectively
3. **Element-wise multiplication**: Preserves dimensionality (768→768), unlike concatenation which changes feature space
4. **Gate overhead**: Only ~100k additional parameters (~0.08% of RoBERTa) for the gate network

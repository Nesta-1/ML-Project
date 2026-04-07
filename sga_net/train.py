"""
SGA-Net Unified Training Script
---------------------------------
Trains any of the 3 models: vanilla, concat, sga
Supports mixed precision, gradient accumulation, early stopping.
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from datasets import load_from_disk

from baseline_roberta import VanillaRoberta
from concat_model import ConcatFusionRoberta
from sga_model import SentimentGatedRoberta

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
MODEL_REGISTRY = {
    "vanilla": VanillaRoberta,
    "concat": ConcatFusionRoberta,
    "sga": SentimentGatedRoberta,
}


def set_seed(seed: int = SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Custom collate to handle 4-dim sentiment features."""
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    # Stack 4 VADER features: compound, pos, neg, neu
    sentiment_score = torch.stack([
        torch.tensor([
            x["sentiment_compound"],
            x["sentiment_pos"],
            x["sentiment_neg"],
            x["sentiment_neu"],
        ], dtype=torch.float32) for x in batch
    ])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sentiment_score": sentiment_score,  # (batch, 4)
    }


def create_dataloaders(data_dir: str, batch_size: int, max_samples: int = None):
    """Load Arrow dataset and create DataLoaders."""
    dataset = load_from_disk(data_dir)

    if max_samples:
        for split in dataset:
            n = min(max_samples, len(dataset[split]))
            dataset[split] = dataset[split].select(range(n))

    train_loader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        dataset["validation"], batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        dataset["test"], batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sentiment_score = batch["sentiment_score"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                out = model(input_ids, attention_mask, sentiment_score=sentiment_score, labels=labels)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids, attention_mask, sentiment_score=sentiment_score, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % 50 == 0:
            print(f"    Step {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sentiment_score = batch["sentiment_score"].to(device)

        out = model(input_ids, attention_mask, sentiment_score=sentiment_score, labels=labels)
        total_loss += out["loss"].item()

        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────────────────────────────────────

def train_model(args):
    """Main training pipeline."""
    set_seed(SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training: {args.model.upper()} model")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.max_samples
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Model
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Mixed precision
    scaler = GradScaler() if device.type == "cuda" else None

    # Output dir
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_f1 = 0.0
    patience_counter = 0
    patience = args.patience

    for epoch in range(args.epochs):
        print(f"\n── Epoch {epoch+1}/{args.epochs} ──")
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Save best model
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"  Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ Saved best model (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  ⚠ Early stopping triggered after {epoch+1} epochs")
                break

    # Save final model and history
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation
    print(f"\n{'='*60}")
    print("  Final Test Evaluation")
    print(f"{'='*60}")
    
    # Load best model for test
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Macro F1: {test_f1:.4f}")

    # Save test predictions
    np.savez(
        output_dir / "test_predictions.npz",
        preds=test_preds, labels=test_labels
    )

    results = {
        "model": args.model,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_loss": float(test_loss),
        "best_val_f1": float(best_val_f1),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Training complete. Results saved to {output_dir}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGA-Net Training")
    parser.add_argument("--model", type=str, required=True,
                        choices=["vanilla", "concat", "sga"],
                        help="Model to train")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                        help="Path to Arrow dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per split (for debugging)")
    args = parser.parse_args()

    train_model(args)

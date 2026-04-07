"""
SGA-Net Data Pipeline
---------------------
Loads Kaggle Cyberbullying Dataset, cleans text, computes VADER sentiment,
splits 80/10/10, tokenizes with roberta-base, saves as Arrow format.
"""

import re
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import contractions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
MAX_LENGTH = 128
MODEL_NAME = "roberta-base"

LABEL_MAP = {
    "age": 0,
    "ethnicity": 1,
    "gender": 2,
    "not_cyberbullying": 3,
    "other_cyberbullying": 4,
    "religion": 5,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# ──────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean tweet text:
    - Remove @handles
    - Remove URLs
    - Expand contractions
    - Keep emojis and stopwords
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    # Remove @handles
    text = re.sub(r"@\w+", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Expand contractions
    text = contractions.fix(text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment Analysis
# ──────────────────────────────────────────────────────────────────────────────

def compute_sentiment(texts: list[str]) -> np.ndarray:
    """
    Compute VADER sentiment features.
    Returns 4 features per text: [compound_norm, pos, neg, neu]
    compound ∈ [-1, 1] → normalized = (compound + 1) / 2
    pos, neg, neu are already in [0, 1]
    """
    analyzer = SentimentIntensityAnalyzer()
    all_scores = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        compound_norm = (scores["compound"] + 1.0) / 2.0  # Map [-1,1] → [0,1]
        all_scores.append([
            compound_norm,
            scores["pos"],
            scores["neg"],
            scores["neu"],
        ])
    return np.array(all_scores, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(csv_path: str, output_dir: str, max_samples: int = None):
    """Complete data processing pipeline."""
    
    print(f"[1/6] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"       Loaded {len(df)} samples")
    
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=SEED).reset_index(drop=True)
        print(f"       Subsampled to {len(df)} samples")

    # ── Clean text ──
    print("[2/6] Cleaning text...")
    df["clean_text"] = df["tweet_text"].apply(clean_text)
    
    # Drop empty texts
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"       {len(df)} samples after cleaning")
    
    # ── Encode labels ──
    print("[3/6] Encoding labels...")
    df["label"] = df["cyberbullying_type"].map(LABEL_MAP)
    
    # Verify no NaN labels
    assert df["label"].notna().all(), "Found unmapped labels!"
    df["label"] = df["label"].astype(int)

    # ── Sentiment scores (4 features: compound, pos, neg, neu) ──
    print("[4/6] Computing VADER sentiment features (4-dim)...")
    sentiment_array = compute_sentiment(df["clean_text"].tolist())
    df["sentiment_compound"] = sentiment_array[:, 0]
    df["sentiment_pos"] = sentiment_array[:, 1]
    df["sentiment_neg"] = sentiment_array[:, 2]
    df["sentiment_neu"] = sentiment_array[:, 3]
    
    print(f"       Compound stats: mean={df['sentiment_compound'].mean():.3f}, "
          f"std={df['sentiment_compound'].std():.3f}")
    print(f"       Pos/Neg/Neu means: {df['sentiment_pos'].mean():.3f} / "
          f"{df['sentiment_neg'].mean():.3f} / {df['sentiment_neu'].mean():.3f}")
    
    # ── Train/Val/Test split (80/10/10) ──
    print("[5/6] Splitting dataset (80/10/10)...")
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"]
    )
    
    print(f"       Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ── Tokenize and save ──
    print("[6/6] Tokenizing and saving Arrow format...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    def make_hf_dataset(split_df: pd.DataFrame) -> Dataset:
        sentiment_cols = ["sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu"]
        ds = Dataset.from_pandas(split_df[["clean_text", "label"] + sentiment_cols].reset_index(drop=True))
        
        def tokenize_fn(examples):
            enc = tokenizer(
                examples["clean_text"],
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )
            return enc
        
        ds = ds.map(tokenize_fn, batched=True, desc="Tokenizing")
        ds = ds.rename_column("label", "labels")
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels",
                                         "sentiment_compound", "sentiment_pos",
                                         "sentiment_neg", "sentiment_neu"])
        return ds

    dataset_dict = DatasetDict({
        "train": make_hf_dataset(train_df),
        "validation": make_hf_dataset(val_df),
        "test": make_hf_dataset(test_df),
    })
    
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"  Train: {len(dataset_dict['train'])} samples")
    print(f"  Val:   {len(dataset_dict['validation'])} samples")
    print(f"  Test:  {len(dataset_dict['test'])} samples")
    print(f"  Classes: {list(LABEL_MAP.keys())}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Tokenizer: {MODEL_NAME}")
    print(f"{'='*60}")
    
    return dataset_dict


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGA-Net Data Pipeline")
    parser.add_argument("--csv", type=str, default="../cyberbullying_tweets.csv",
                        help="Path to cyberbullying_tweets.csv")
    parser.add_argument("--output", type=str, default="./data/processed",
                        help="Output directory for Arrow dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (for debugging)")
    args = parser.parse_args()
    
    run_pipeline(args.csv, args.output, args.max_samples)

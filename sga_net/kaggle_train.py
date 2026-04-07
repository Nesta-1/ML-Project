"""
SGA-Net Full Training Script for Kaggle
========================================
Upload this file + your sga_net/ folder + cyberbullying_tweets.csv to Kaggle.
Enable GPU accelerator (T4), then run all cells.

Estimated time: ~45 minutes total for 3 models on T4 GPU.

Instructions:
1. Create a new Kaggle Notebook
2. Upload cyberbullying_tweets.csv as a dataset
3. Upload the sga_net/ folder (or copy all .py files)
4. Enable GPU: Settings → Accelerator → GPU T4 x2
5. Paste this script in a code cell and run
"""

import os
import subprocess
import sys
import time

# ──────────────────────────────────────────────────────────────────────────────
# Setup: Install dependencies
# ──────────────────────────────────────────────────────────────────────────────

print(">>> Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.30.0", "datasets>=2.14.0", "accelerate>=0.21.0",
    "vaderSentiment>=3.3.2", "contractions>=0.1.73", "emoji>=2.8.0"
])

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — adjust paths based on your Kaggle setup
# ──────────────────────────────────────────────────────────────────────────────

# If you uploaded files to /kaggle/input/your-dataset-name/
# adjust these paths accordingly
CSV_PATH = "/kaggle/input/cyberbullying-tweets/cyberbullying_tweets.csv"
SGA_NET_DIR = "/kaggle/working/sga_net"

# Check if CSV exists at expected path, try alternatives
if not os.path.exists(CSV_PATH):
    # Try common alternatives
    alternatives = [
        "/kaggle/input/cyberbullying-classification/cyberbullying_tweets.csv",
        "/kaggle/input/cyberbullying_tweets.csv",
        "./cyberbullying_tweets.csv",
    ]
    for alt in alternatives:
        if os.path.exists(alt):
            CSV_PATH = alt
            break
    else:
        # List available input files
        print("Available input files:")
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        raise FileNotFoundError("Could not find cyberbullying_tweets.csv. Update CSV_PATH.")

print(f"Using CSV: {CSV_PATH}")
print(f"SGA-Net dir: {SGA_NET_DIR}")

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Data Pipeline
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1: DATA PIPELINE")
print("="*60)

os.chdir(SGA_NET_DIR)
t0 = time.time()

subprocess.check_call([
    sys.executable, "data_pipeline.py",
    "--csv", CSV_PATH,
    "--output", "./data/processed"
])

print(f"Data pipeline completed in {time.time()-t0:.1f}s")

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Train all 3 models
# ──────────────────────────────────────────────────────────────────────────────

MODELS = ["vanilla", "concat", "sga"]
EPOCHS = 10
BATCH_SIZE = 16
PATIENCE = 3
LR = 2e-5

for model_name in MODELS:
    print("\n" + "="*60)
    print(f"STEP 2: TRAINING {model_name.upper()}")
    print("="*60)
    
    t0 = time.time()
    
    subprocess.check_call([
        sys.executable, "train.py",
        "--model", model_name,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--patience", str(PATIENCE),
        "--lr", str(LR),
        "--data_dir", "./data/processed",
        "--output_dir", "./checkpoints",
    ])
    
    elapsed = time.time() - t0
    print(f"\n{model_name.upper()} training completed in {elapsed/60:.1f} minutes")

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluation
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 3: EVALUATION")
print("="*60)

subprocess.check_call([
    sys.executable, "evaluate.py",
    "--checkpoint_dir", "./checkpoints",
    "--output_dir", "./results"
])

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Display Results
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

import json

results_path = "./results/comparison_results.json"
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}")
    
    # Highlight SGA gate alpha if available
    sga_checkpoint = "./checkpoints/sga/best_model.pt"
    if os.path.exists(sga_checkpoint):
        import torch
        state = torch.load(sga_checkpoint, map_location="cpu")
        if "gate_alpha" in state:
            print(f"\nSGA Gate Alpha (learned): {state['gate_alpha'].item():.4f}")
    
    print("\n✓ All steps completed successfully!")
    print(f"Results saved to: {os.path.abspath('./results/')}")
    print(f"Checkpoints saved to: {os.path.abspath('./checkpoints/')}")
else:
    print("⚠ Results file not found!")

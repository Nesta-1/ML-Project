"""Quick environment and dataset check."""
import pandas as pd

# Dataset info
df = pd.read_csv("../cyberbullying_tweets.csv")
print("=== Dataset Info ===")
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:")
print(df["cyberbullying_type"].value_counts().to_string())
print(f"\nUnique classes: {df['cyberbullying_type'].unique().tolist()}")

# Check dependencies
deps = {
    "torch": False,
    "transformers": False,
    "datasets": False,
    "vaderSentiment": False,
    "plotly": False,
    "sklearn": False,
}

for dep in deps:
    try:
        __import__(dep)
        deps[dep] = True
    except ImportError:
        pass

print("\n=== Dependencies ===")
for dep, available in deps.items():
    print(f"  {dep}: {'OK' if available else 'MISSING'}")

try:
    import torch
    print(f"\n  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except:
    pass

# Run Full SGA-Net Pipeline
# This script runs data processing, training for all models, and evaluation.
# Usage: ./run_full_training.ps1

# Write-Host ">>> Step 1: Data Pipeline"
# python data_pipeline.py --csv ../cyberbullying_tweets.csv --output ./data/processed
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">>> Step 2: Training Vanilla RoBERTa"
python train.py --model vanilla --epochs 3 --batch_size 16 --max_samples 2000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">>> Step 3: Training Concat Fusion"
python train.py --model concat --epochs 3 --batch_size 16 --max_samples 2000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">>> Step 4: Training SGA-RoBERTa"
python train.py --model sga --epochs 3 --batch_size 16 --max_samples 2000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">>> Step 5: Evaluation"
python evaluate.py --checkpoint_dir ./checkpoints --output_dir ./results
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">>> All steps completed successfully!"

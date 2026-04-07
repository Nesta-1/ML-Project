
import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
import sys

# Add parent directory to path to import baselines modules if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baselines.data_utils import prepare_data
from baselines.models import get_models

def main():
    parser = argparse.ArgumentParser(description="Run baseline models for cyberbullying detection.")
    parser.add_argument("--data_path", type=str, default="cyberbullying_tweets.csv", help="Path to the dataset.")
    parser.add_argument("--no_smote", action="store_true", help="Disable SMOTE resampling.")
    parser.add_argument("--no_sa", action="store_true", help="Disable Sentiment Analysis filtering.")
    parser.add_argument("--output_file", type=str, default="baselines/results.txt", help="File to save results.")
    args = parser.parse_args()

    # Configuration based on arguments
    use_smote = not args.no_smote
    use_sa_filter = not args.no_sa
    
    print(f"Configuration: SMOTE={use_smote}, BV-SA Filtering={use_sa_filter}")

    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"Error: Dataset not found at {args.data_path}")
        return

    # Prepare Data
    try:
        X_train, X_test, y_train, y_test, vectorizer, class_map = prepare_data(
            args.data_path, 
            use_sentiment_filter=use_sa_filter, 
            apply_smote=use_smote
        )
    except Exception as e:
        print(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get Models
    models = get_models()
    
    results = []
    
    print("\n" + "="*50)
    print("Starting Training and Evaluation")
    print("="*50)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score (Weighted): {f1:.4f}")
            print(f"  Time taken: {train_time:.2f}s")
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "F1 Score": f1,
                "Time (s)": train_time
            })
            
            # Save detailed report to file
            with open(args.output_file, "a") as f:
                f.write(f"\n{'='*30}\n")
                f.write(f"Model: {name}\n")
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Time: {train_time:.2f}s\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, y_pred, target_names=list(class_map.keys())))
                f.write("\n")
                
        except Exception as e:
            print(f"  Error training {name}: {e}")
            
    # Summary
    print("\n" + "="*50)
    print("Summary of Results")
    print("="*50)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy", ascending=False)
    print(results_df)
    
    # Save summary to file
    with open(args.output_file, "a") as f:
        f.write("\n" + "="*50 + "\n")
        f.write("Summary Results\n")
        f.write(results_df.to_string())
        f.write("\n")

if __name__ == "__main__":
    main()

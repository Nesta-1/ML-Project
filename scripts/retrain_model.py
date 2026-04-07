import sys
import os
import joblib
from unittest.mock import MagicMock

# Mock streamlit
sys.modules['streamlit'] = MagicMock()
class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    def __setattr__(self, key, value):
        self[key] = value
sys.modules['streamlit'].session_state = SessionState()
# Mock columns to return iterable
def mock_columns(spec, *args, **kwargs):
    if isinstance(spec, int):
        return [MagicMock() for _ in range(spec)]
    return [MagicMock() for _ in spec]
sys.modules['streamlit'].columns = mock_columns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app2 import CyberbullyingClassifier

def main():
    print("Retraining model with updated settings (len > 0, extra synthetic data)...")
    
    # Initialize classifier
    classifier = CyberbullyingClassifier()
    
    # Path to model file
    model_path = 'cyberguard_model.joblib'
    
    # Move existing model to backup just in case
    if os.path.exists(model_path):
        os.rename(model_path, model_path + '.bak')
        print(f"Backed up existing model to {model_path}.bak")
    
    # Force training
    print("Training model...")
    # Passing use_dataset=False trains on synthetic data which we improved
    train_acc, test_acc = classifier.train_model(use_dataset=False)
    
    if train_acc is not None:
        print(f"Training complete. Accuracy: Train={train_acc:.4f}, Test={test_acc:.4f}")
        
        # Save model
        print(f"Saving model to {model_path}...")
        joblib.dump({
            'model': classifier.model,
            'vectorizer': classifier.vectorizer,
        }, model_path)
        print("Model saved successfully.")
    else:
        print("Training failed.")

if __name__ == "__main__":
    main()

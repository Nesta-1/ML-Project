import sys
import os
import joblib
from unittest.mock import MagicMock

# Mock streamlit before importing app2
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

# Add parent directory to path to import app2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app2 import CyberbullyingClassifier

def main():
    print("Verifying 0.3 Threshold and Preprocessing Fixes...")
    
    classifier = CyberbullyingClassifier()
    model_path = os.path.join(os.getcwd(), 'cyberguard_model.joblib')
    
    # Manually load model since app2 logic might rely on session state or file existence in specific way
    # But app2.CyberbullyingClassifier has methods that auto-load or we can set it manually
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        classifier.model = data['model']
        classifier.vectorizer = data['vectorizer']
        classifier.is_trained = True
    else:
        print("Model file not found!")
        return

    test_cases = [
        "fk you",
        "fuck you",
        "kill yourself",
        "hello world",
        "you are amazing"
    ]
    
    print(f"{'Text':<20} | {'Prediction':<20} | {'Confidence':<10} | {'Tokens':<20}")
    print("-" * 80)
    
    for text in test_cases:
        prediction, confidence, _, tokens = classifier.predict(text)
        print(f"{text:<20} | {prediction:<20} | {confidence:.4f}     | {str(tokens):<20}")

if __name__ == "__main__":
    main()

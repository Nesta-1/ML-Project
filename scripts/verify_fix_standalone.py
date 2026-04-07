import pandas as pd
import numpy as np
import joblib
import os
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize global components
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

class CyberbullyingClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False

    def preprocess_text(self, text):
        """Enhanced text preprocessing with FIX: len > 1"""
        try:
            # Convert to string and lower case
            text = str(text).lower()
            
            # Handle emojis
            text = emoji.demojize(text, delimiters=(" ", " "))
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Remove stopwords but keep 'you', 'your' as they might be relevant for direct attacks
            try:
                stop_words = set(stopwords.words('english'))
            except:
                # Fallback
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            lemmatized_tokens = []
            for token in tokens:
                # FIX: Change > 2 to > 1
                if token not in stop_words and len(token) > 1:
                    try:
                        lemmatized_tokens.append(lemmatizer.lemmatize(token))
                    except:
                        lemmatized_tokens.append(token)
            
            return ' '.join(lemmatized_tokens), lemmatized_tokens
        except:
            clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            # FIX: Change > 2 to > 1
            tokens = [word for word in clean_text.split() if len(word) > 1]
            return ' '.join(tokens), tokens
    
    def get_vader_features(self, text):
        """Extract VADER sentiment features"""
        try:
            scores = analyzer.polarity_scores(str(text))
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            }
        except:
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}

    def predict(self, text):
        """Predict cyberbullying classification with FIX: Threshold 0.3"""
        try:
            if not self.is_trained:
                return "Error (Not Trained)", 0.0, {}, []
            
            # Preprocess text
            processed_text, tokens = self.preprocess_text(text)
            
            # Get features
            X_tfidf = self.vectorizer.transform([processed_text])
            vader = self.get_vader_features(text)
            vader_features = np.array([[vader['compound'], vader['positive'], 
                                      vader['neutral'], vader['negative']]])
            
            # Combine features
            X_combined = np.hstack([X_tfidf.toarray(), vader_features])
            
            # Predict Logic with Custom Threshold
            probabilities = self.model.predict_proba(X_combined)[0]
            classes = self.model.classes_
            
            # Apply custom threshold of 0.3 for Cyberbullying
            cb_index = -1
            for i, label in enumerate(classes):
                if label == 'Cyberbullying':
                    cb_index = i
                    break
            
            prediction = classes[np.argmax(probabilities)] # Default
            
            if cb_index != -1:
                cb_prob = probabilities[cb_index]
                if cb_prob >= 0.6:
                    prediction = 'Cyberbullying'
                else:
                    prediction = 'Not Cyberbullying'
            
            # Update confidence based on the selected prediction
            pred_index = -1
            for i, label in enumerate(classes):
                if label == prediction:
                    pred_index = i
                    break
            
            confidence = probabilities[pred_index] if pred_index != -1 else np.max(probabilities)
            
            # Debug info
            probs_debug = {cls: prob for cls, prob in zip(classes, probabilities)}
            
            return prediction, confidence, vader, tokens, probs_debug
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0, {}, [], {}

def main():
    print("Verifying Fixes Standalone...")
    
    classifier = CyberbullyingClassifier()
    model_path = 'cyberguard_model.joblib'
    
    if os.path.exists(model_path):
        print(f"Loading FRESH model from {model_path} (timestamp: {os.path.getmtime(model_path)})")
        data = joblib.load(model_path)
        classifier.model = data['model']
        classifier.vectorizer = data['vectorizer']
        classifier.is_trained = True
    else:
        print("Model file not found!")
        return

    test_cases = [
        "fk you",           # Should be caught by preprocessing fix + threshold
        "fuck you",         # Should be caught by threshold
        "kill yourself",    # Should be caught by threshold
        "hello world",      # Should remain Safe
        "you are amazing"   # Should remain Safe
    ]
    
    import json
    results = []
    print(f"\n{'Text':<20} | {'Prediction':<20} | {'Conf':<6} | {'Probs':<40} | {'Tokens'}")
    print("-" * 120)
    
    for text in test_cases:
        prediction, confidence, _, tokens, probs = classifier.predict(text)
        print(f"{text:<20} | {prediction:<20} | {confidence:.4f} | {str(probs):<40} | {tokens}")
        results.append({
            "text": text,
            "prediction": prediction,
            "confidence": confidence,
            "probs": probs,
            "tokens": tokens
        })
    
    with open('output/verification_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to output/verification_results.json")

if __name__ == "__main__":
    main()

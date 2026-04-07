import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import joblib
import emoji
import json
import sys

# --- Replicated Logic from app2.py ---

# Initialize components
analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

def download_nltk_data():
    import sys
    import io
    required_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for item in required_downloads:
        try:
            # Suppress NLTK's stderr
            _old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                nltk.download(item, quiet=True)
            finally:
                sys.stderr = _old_stderr
        except Exception:
            pass

download_nltk_data()

class CyberbullyingClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.training_data = None
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        try:
            text = str(text).lower() # Ensure string
            text = emoji.demojize(text, delimiters=(" ", " "))
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            lemmatized_tokens = []
            for token in tokens:
                if token not in stop_words and len(token) > 2:
                    try:
                        lemmatized_tokens.append(lemmatizer.lemmatize(token))
                    except:
                        lemmatized_tokens.append(token)
            
            return ' '.join(lemmatized_tokens), lemmatized_tokens
        except:
            clean_text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
            tokens = [word for word in clean_text.split() if len(word) > 2]
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

    def predict_proba(self, text):
        """Predict cyberbullying classification probability (Replicated from app2.py predict but returns dict)"""
        # (This is a slightly modified version of predict to return raw data for analysis)
        try:
            if not self.is_trained:
                return None
            
            # Preprocess text
            processed_text, tokens = self.preprocess_text(text)
            
            # Get features
            X_tfidf = self.vectorizer.transform([processed_text])
            vader = self.get_vader_features(text)
            vader_features = np.array([[vader['compound'], vader['positive'], 
                                      vader['neutral'], vader['negative']]])
            
            # Combine features
            X_combined = np.hstack([X_tfidf.toarray(), vader_features])
            
            # Predict
            prediction = self.model.predict(X_combined)[0]
            probabilities = self.model.predict_proba(X_combined)[0] # [prob_class0, prob_class1]
            classes = self.model.classes_
            
            prob_dict = {class_: prob for class_, prob in zip(classes, probabilities)}
            
            return {
                "text": text,
                "processed_text": processed_text,
                "prediction": prediction,
                "probabilities": prob_dict,
                "vader_scores": vader,
                "tokens": tokens
            }
            
        except Exception as e:
            print(f"Error predicting '{text}': {e}")
            return None

    def load_model(self, path):
        if os.path.isfile(path):
            try:
                data = joblib.load(path)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
                self.is_trained = True
                print(f"Model loaded from {path}")
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
                return False
        else:
            print(f"Model file not found at {path}")
            return False

    def _load_csv_to_dataframe(self, filepath):
        """Load a CSV and return a DataFrame with columns text, label; or None if unsupported format."""
        try:
            # Try reading with different encodings just in case
            try:
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='latin1')

            # classified_tweets.csv: text, suspicious, cyberbullying, hate, suicidal
            if 'text' in df.columns and 'cyberbullying' in df.columns and 'hate' in df.columns and 'suicidal' in df.columns:
                harmful = (df['cyberbullying'].fillna(0) >= 1) | (df['hate'].fillna(0) >= 1) | (df['suicidal'].fillna(0) >= 1)
                df = df.copy()
                df['label'] = harmful.map({True: 'Cyberbullying', False: 'Not Cyberbullying'})
                return df[['text', 'label']].dropna(subset=['text'])
            
            return None
        except Exception as e:
            print(f"Error loading CSV {filepath}: {e}")
            return None

# --- Analysis Logic ---

def main():
    print("Starting Model Analysis...")
    
    # 1. Load Model
    classifier = CyberbullyingClassifier()
    model_path = os.path.join(os.getcwd(), 'cyberguard_model.joblib')
    if not classifier.load_model(model_path):
        print("CRITICAL: Could not load model. Ensure 'cyberguard_model.joblib' exists.")
        return

    # 4. Investigate Specific Cases (MOVED TO TOP)
    print("\n--- Investigating Specific Cases ---")
    
    test_cases = [
        "fk you",
        "fuck you",
        "you are invalid",
        "kill yourself",
        "you are a waste of space",
        "you fail at life",
        "hello world",
        "you are amazing",
        "i hate you",
        "i dislike you",
        "this is bad"
    ]
    
    case_results = []
    
    for text in test_cases:
        result = classifier.predict_proba(text)
        if result:
            print(f"Text: '{text}' -> {result['prediction']} (Conf: {max(result['probabilities'].values()):.4f})")
            print(f"   Probs: {result['probabilities']}")
            print(f"   VADER: {result['vader_scores']}")
            print(f"   Processed: '{result['processed_text']}'")
            case_results.append(result)
            
    with open('output/specific_cases.json', 'w') as f:
        json.dump(case_results, f, indent=4)

    # 2. Load Data for Confusion Matrix
    csv_path = os.path.join(os.getcwd(), 'classified_tweets.csv')
    df = classifier._load_csv_to_dataframe(csv_path)
    
    if df is not None:
        print(f"Loaded {len(df)} samples from {csv_path}")
        
        # Sample for speed (2000 samples)
        sample_size = min(len(df), 2000)
        df = df.sample(sample_size, random_state=42) 
        print(f"Sampling {len(df)} rows for Confusion Matrix generation...")

        y_true = df['label'].tolist()
        y_pred = []
        
        print("Running predictions on dataset...")
        # Dictionary to store failure cases
        failures = []
        
        for index, row in df.iterrows():
            if index % 100 == 0:
                print(f"Processed {index}/{len(df)}...")
            
            result = classifier.predict_proba(row['text'])
            if result:
                y_pred.append(result['prediction'])
                
                if result['prediction'] != row['label']:
                    failures.append({
                        "text": row['text'],
                        "true_label": row['label'],
                        "predicted_label": result['prediction'],
                        "probabilities": result['probabilities'],
                        "vader": result['vader_scores']
                    })
            else:
                y_pred.append("Error")

        # 3. Generate Confusion Matrix
        # Filter out errors
        valid_indices = [i for i, p in enumerate(y_pred) if p != "Error"]
        y_true_clean = [y_true[i] for i in valid_indices]
        y_pred_clean = [y_pred[i] for i in valid_indices]
        
        if not y_true_clean:
             print("No valid predictions generated.")
             return

        labels = sorted(list(set(y_true_clean)))
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('output/confusion_matrix.png')
        print("Saved confusion matrix to output/confusion_matrix.png")
        
        # Classification Report
        report = classification_report(y_true_clean, y_pred_clean, output_dict=True)
        with open('output/classification_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        # Save failures heavily weighted by confidence to see high confidence errors
        failures.sort(key=lambda x: max(x['probabilities'].values()) if x['probabilities'] else 0, reverse=True)
        with open('output/failure_cases.json', 'w') as f:
            json.dump(failures[:100], f, indent=4) # Top 100 failures
            
    else:
        print("Could not load dataset for Confusion Matrix.")


if __name__ == "__main__":
    main()

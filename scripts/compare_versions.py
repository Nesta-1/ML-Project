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
import json

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

class ModelWrapper:
    def __init__(self, model_path, threshold, min_token_len):
        self.model_path = model_path
        self.threshold = threshold
        self.min_token_len = min_token_len
        self.vectorizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
        else:
            print(f"Error: Model file {self.model_path} not found.")

    def preprocess_text(self, text):
        try:
            text = str(text).lower()
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
                # DYNAMIC TOKEN LENGTH LOGIC
                if token not in stop_words and len(token) > self.min_token_len:
                    try:
                        lemmatized_tokens.append(lemmatizer.lemmatize(token))
                    except:
                        lemmatized_tokens.append(token)
            
            return ' '.join(lemmatized_tokens)
        except:
            return ""
    
    def get_vader_features(self, text):
        try:
            scores = analyzer.polarity_scores(str(text))
            return [scores['compound'], scores['pos'], scores['neu'], scores['neg']]
        except:
            return [0.0, 0.0, 1.0, 0.0]

    def predict(self, text):
        if self.model is None:
            return "Error", 0.0
        
        processed_text = self.preprocess_text(text)
        X_tfidf = self.vectorizer.transform([processed_text])
        vader = self.get_vader_features(text)
        vader_features = np.array([vader])
        X_combined = np.hstack([X_tfidf.toarray(), vader_features])
        
        probabilities = self.model.predict_proba(X_combined)[0]
        classes = self.model.classes_
        
        cb_index = -1
        for i, label in enumerate(classes):
            if label == 'Cyberbullying':
                cb_index = i
                break
        
        prediction = classes[np.argmax(probabilities)] # Default
        confidence = np.max(probabilities)
        prob_cb = 0.0
        
        if cb_index != -1:
            prob_cb = probabilities[cb_index]
            # DYNAMIC THRESHOLD LOGIC
            if prob_cb >= self.threshold:
                prediction = 'Cyberbullying'
            else:
                prediction = 'Not Cyberbullying'
            
            # Recalculate confidence based on prediction
            if prediction == 'Cyberbullying':
                confidence = prob_cb
            else:
                confidence = 1 - prob_cb
                
        return prediction, prob_cb

def main():
    print("Comparing Initial vs Present Model...")
    
    # 1. Initial Version (Backup model, old logic)
    # Threshold=0.5, Min Token Len > 2
    initial_model = ModelWrapper('cyberguard_model.joblib.bak', threshold=0.5, min_token_len=2)
    
    # 2. Present Version (New model, new logic)
    # Threshold=0.6, Min Token Len > 0 (len >= 1 is actually > 0 in updated code logic checks "len > 0")
    present_model = ModelWrapper('cyberguard_model.joblib', threshold=0.6, min_token_len=0) 
    
    test_cases = [
        "fk you",
        "fuck you",
        "kill yourself",
        "hello world",
        "you are amazing",
        "u r stupid",
        "go die",
        "have a nice day"
    ]
    
    results = []
    
    print(f"\n{'Text':<20} | {'Initial (Pred)':<20} | {'Prob(CB)':<8} | {'Present (Pred)':<20} | {'Prob(CB)':<8}")
    print("-" * 100)
    
    for text in test_cases:
        pred_old, prob_old = initial_model.predict(text)
        pred_new, prob_new = present_model.predict(text)
        
        print(f"{text:<20} | {pred_old:<20} | {prob_old:.4f}   | {pred_new:<20} | {prob_new:.4f}")
        
        results.append({
            "text": text,
            "initial": {"prediction": pred_old, "prob_cb": prob_old},
            "present": {"prediction": pred_new, "prob_cb": prob_new},
            "change": prob_new - prob_old
        })
        
    with open('output/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()

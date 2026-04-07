import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import emoji
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize components
analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Mocking the class logic from app2.py for standalone testing
class CyberbullyingClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        try:
            text = text.lower()
            text = emoji.demojize(text, delimiters=(" ", " "))
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = text.split()
            processed_tokens = []
            for token in tokens:
                if len(token) > 0:
                    processed_tokens.append(lemmatizer.lemmatize(token))
            return ' '.join(processed_tokens), processed_tokens
        except:
            return text, []

    def get_vader_features(self, text):
        try:
            scores = analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            }
        except:
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}

    def train_model(self):
        # Improved synthetic data with specific labels
        synthetic_data = [
            # Gender
            ("you are a bitch", "gender"),
            ("sexist pig", "gender"),
            ("go back to the kitchen", "gender"),
            ("gay", "gender"),
            
            # Religion
            ("terrorist", "religion"),
            ("muslims are bad", "religion"),
            ("jewish scum", "religion"),
            
            # Ethnicity
            ("white trash", "ethnicity"),
            ("black monkeys", "ethnicity"),
            ("nigger", "ethnicity"),
            
            # Age
            ("kid", "age"),
            ("school shooter", "age"),
            ("middle school loser", "age"),
            
            # Other Cyberbullying
            ("you're such an idiot", "other_cyberbullying"),
            ("kill yourself", "other_cyberbullying"),
            ("you're ugly and stupid", "other_cyberbullying"),
            ("shut up loser", "other_cyberbullying"),
            ("fk you", "other_cyberbullying"),
            ("fuck you", "other_cyberbullying"),
            ("u r stupid", "other_cyberbullying"),
            
            # Not Cyberbullying
            ("great job on the project", "not_cyberbullying"),
            ("thank you for your help", "not_cyberbullying"),
            ("have a wonderful day", "not_cyberbullying"),
            ("congratulations", "not_cyberbullying"),
            ("you look nice today", "not_cyberbullying"),
            ("good morning", "not_cyberbullying"),
            ("the meeting is tomorrow", "not_cyberbullying"),
            ("what time is it", "not_cyberbullying"),
        ] * 50
        
        texts = [item[0] for item in synthetic_data]
        labels = [item[1] for item in synthetic_data]
        
        processed_texts = []
        for text in texts:
            processed_text, _ = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        
        self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 3))
        X_tfidf = self.vectorizer.fit_transform(processed_texts)
        
        vader_features = []
        for text in texts:
            vader = self.get_vader_features(text)
            vader_features.append([vader['compound'], vader['positive'], 
                                 vader['neutral'], vader['negative']])
        
        X_combined = np.hstack([X_tfidf.toarray(), np.array(vader_features)])
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_combined, y_encoded)
        
        self.model = ExtraTreesClassifier(n_estimators=200, random_state=42)
        self.model.fit(X_train_balanced, y_train_balanced)
        self.is_trained = True
        print("Model trained on 6 classes.")

    def predict(self, text):
        if not self.is_trained:
            self.train_model()
            
        processed_text, _ = self.preprocess_text(text)
        X_tfidf = self.vectorizer.transform([processed_text])
        vader = self.get_vader_features(text)
        vader_features = np.array([[vader['compound'], vader['positive'], 
                                  vader['neutral'], vader['negative']]])
        X_combined = np.hstack([X_tfidf.toarray(), vader_features])
        
        probabilities = self.model.predict_proba(X_combined)[0]
        class_labels = self.label_encoder.classes_
        
        # Find not_cyberbullying index
        not_cb_index = -1
        for i, label in enumerate(class_labels):
            if label == 'not_cyberbullying':
                not_cb_index = i
                break
        
        if not_cb_index != -1:
            prob_safe = probabilities[not_cb_index]
            prob_bullying = 1.0 - prob_safe
        else:
            prob_bullying = np.max(probabilities)
            prob_safe = 1.0 - prob_bullying
            
        THRESHOLD = 0.60
        specific_category = None
        
        if prob_bullying > THRESHOLD:
            prediction = "Cyberbullying"
            
            # Identify specific type
            probs_copy = probabilities.copy()
            if not_cb_index != -1:
                probs_copy[not_cb_index] = -1.0
            
            max_idx = np.argmax(probs_copy)
            specific_category = class_labels[max_idx]
        else:
            prediction = "Not Cyberbullying"
        
        return prediction, specific_category, prob_bullying

import json
import os

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Run Verification
classifier = CyberbullyingClassifier()
classifier.train_model()

test_cases = [
    ("gay", "gender"),
    ("terrorist", "religion"),
    ("nigger", "ethnicity"),
    ("kill yourself", "other_cyberbullying"),
    ("fk you", "other_cyberbullying"),
    ("hello world", "not_cyberbullying"),
    ("school shooter", "age"),
    ("you are a bitch", "gender")
]

results = []
output_lines = []
output_lines.append("\n--- Verification Results ---")
output_lines.append(f"{'Text':<20} | {'Expected':<20} | {'Predicted':<20} | {'Category':<20} | {'Prob(Bullying)':<15}")
output_lines.append("-" * 105)

for text, expected_category in test_cases:
    pred, category, prob = classifier.predict(text)
    
    # Check if category matches or if it's safe
    match_status = "PASS"
    if expected_category == "not_cyberbullying":
        if pred == "Not Cyberbullying":
            res_category = "N/A"
        else:
            match_status = "FAIL"
            res_category = category
    else:
        if pred == "Cyberbullying" and category == expected_category:
            res_category = category
        elif pred == "Cyberbullying":
            match_status = f"MISMATCH ({category})"
            res_category = category
        else:
            match_status = "FAIL (False Neg)"
            res_category = "N/A"
            
    output_lines.append(f"{text:<20} | {expected_category:<20} | {pred:<20} | {res_category:<20} | {prob:.4f} | {match_status}")
    
    results.append({
        "text": text,
        "expected": expected_category,
        "predicted": pred,
        "category": res_category,
        "probability": prob,
        "status": match_status
    })

# Print to console
for line in output_lines:
    print(line)

# Save to files
with open('output/multiclass_verification.txt', 'w') as f:
    f.write('\n'.join(output_lines))

with open('output/multiclass_verification.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to output/multiclass_verification.txt and output/multiclass_verification.json")

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import emoji

# Initialize components
analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Enhanced text preprocessing"""
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
            if token not in stop_words and len(token) > 2:
                try:
                    lemmatized_tokens.append(lemmatizer.lemmatize(token))
                except:
                    lemmatized_tokens.append(token)
        
        return ' '.join(lemmatized_tokens)
    except:
        return ""

def get_vader_features(text):
    try:
        scores = analyzer.polarity_scores(str(text))
        return [scores['compound'], scores['pos'], scores['neu'], scores['neg']]
    except:
        return [0.0, 0.0, 1.0, 0.0]

def main():
    print("Loading model...")
    try:
        data = joblib.load('cyberguard_model.joblib')
        model = data['model']
        vectorizer = data['vectorizer']
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading dataset...")
    try:
        df = pd.read_csv('classified_tweets.csv')
        # Recreate label logic
        if 'text' in df.columns and 'cyberbullying' in df.columns:
            harmful = (df['cyberbullying'].fillna(0) >= 1) | (df['hate'].fillna(0) >= 1) | (df['suicidal'].fillna(0) >= 1)
            df['label'] = harmful.map({True: 'Cyberbullying', False: 'Not Cyberbullying'})
            df = df.dropna(subset=['text', 'label'])
            
            # Use same sample seed as before to verify
            df = df.sample(min(len(df), 2000), random_state=42)
            
            y_true = df['label'].tolist()
            y_pred = []
            
            print(f"Predicting on {len(df)} samples...")
            
            processed_texts = [preprocess_text(t) for t in df['text']]
            tfidf_features = vectorizer.transform(processed_texts).toarray()
            
            vader_features = np.array([get_vader_features(t) for t in df['text']])
            
            X = np.hstack([tfidf_features, vader_features])
            y_pred = model.predict(X)
            
            labels = ['Cyberbullying', 'Not Cyberbullying']
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            print("\n" + "="*40)
            print("CONFUSION MATRIX VERIFICATION")
            print("="*40)
            print(f"Labels: {labels}")
            print("-" * 20)
            print(f"Row 0 (Actual Cyberbullying):     {cm[0]}")
            print(f"Row 1 (Actual Not Cyberbullying): {cm[1]}")
            print("-" * 20)
            
            tp = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tn = cm[1][1]
            
            print(f"True Positives  (Cyberbullying detected as Cyberbullying): {tp}")
            print(f"False Negatives (Cyberbullying missed/called Safe):      {fn}")
            print(f"False Positives (Safe flagged as Cyberbullying):         {fp}")
            print(f"True Negatives  (Safe correctly identified as Safe):     {tn}")
            print("="*40)
            
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()

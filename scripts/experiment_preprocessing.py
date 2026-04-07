import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import emoji
import sys
import io

# Initialize components
analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

def download_nltk_data():
    required_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for item in required_downloads:
        try:
            _old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                nltk.download(item, quiet=True)
            finally:
                sys.stderr = _old_stderr
        except Exception:
            pass

download_nltk_data()

class ExperimentClassifier:
    def __init__(self, use_stopwords=True, use_lemmatization=True):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.use_stopwords = use_stopwords
        self.use_lemmatization = use_lemmatization
        
    def preprocess_text(self, text):
        """Preprocess text based on configuration"""
        try:
            text = str(text).lower()
            text = emoji.demojize(text, delimiters=(" ", " "))
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Stopword removal logic
            if not self.use_stopwords: # IF use_stopwords is False, we REMOVE them (confusing naming, let's stick to "remove_stopwords")
                 # Wait, let's align with the goal. 
                 # Goal: "remove stop words and Lemmatization if present" -> User asked if removing them makes it MORE effective.
                 # Current app2.py DOES remove them.
                 # App2.py: if token not in stop_words... (Removes them)
                 # Experiment: 
                 #   Config 1 (Baseline): Remove Stopwords = True, Lemmatize = True
                 #   Config 2 (Experimental): Remove Stopwords = False, Lemmatize = False
                 pass

            try:
                 stop_words = set(stopwords.words('english'))
            except:
                 stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

            final_tokens = []
            for token in tokens:
                # Filter stopwords if configured to remove them
                if self.use_stopwords and token in stop_words:
                    continue
                
                # Lemmatize if configured
                if self.use_lemmatization:
                    try:
                        token = lemmatizer.lemmatize(token)
                    except:
                        pass
                
                if len(token) > 1:
                    final_tokens.append(token)
            
            return ' '.join(final_tokens)
        except Exception as e:
            return text

    def get_vader_features(self, text):
        try:
            scores = analyzer.polarity_scores(str(text))
            return [scores['compound'], scores['pos'], scores['neu'], scores['neg']]
        except:
            return [0.0, 0.0, 1.0, 0.0]

    def train_and_evaluate(self, X_train_raw, X_test_raw, y_train, y_test):
        # 1. Preprocess
        print(f"  Preprocessing... (Remove Stopwords={self.use_stopwords}, Lemmatize={self.use_lemmatization})")
        X_train_processed = [self.preprocess_text(text) for text in X_train_raw]
        X_test_processed = [self.preprocess_text(text) for text in X_test_raw]

        # 2. Vectorize
        self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 3))
        X_train_tfidf = self.vectorizer.fit_transform(X_train_processed)
        X_test_tfidf = self.vectorizer.transform(X_test_processed)

        # 3. VADER features
        X_train_vader = np.array([self.get_vader_features(text) for text in X_train_raw])
        X_test_vader = np.array([self.get_vader_features(text) for text in X_test_raw])

        # 4. Combine
        X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_vader])
        X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_vader])

        # 5. SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_combined, y_train)

        # 6. Train
        self.model = ExtraTreesClassifier(n_estimators=100, random_state=42) # Slightly faster for experiment
        self.model.fit(X_train_balanced, y_train_balanced)

        # 7. Evaluate
        y_pred = self.model.predict(X_test_combined)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "matrix": confusion_matrix(y_test, y_pred)
        }

def load_data():
    dfs = []
    for f in ['classified_tweets.csv', 'cyberbullying_tweets.csv']:
        if os.path.exists(f):
            try:
                if f == 'cyberbullying_tweets.csv':
                     df = pd.read_csv(f)
                     df = df.rename(columns={'tweet_text': 'text'})
                     df['label'] = df['cyberbullying_type'].apply(lambda x: 'Not Cyberbullying' if str(x).strip() == 'not_cyberbullying' else 'Cyberbullying')
                     dfs.append(df[['text', 'label']])
                else:
                    df = pd.read_csv(f)
                    if 'text' in df.columns and 'cyberbullying' in df.columns:
                         harmful = (df['cyberbullying'].fillna(0) >= 1) | (df['hate'].fillna(0) >= 1) | (df['suicidal'].fillna(0) >= 1)
                         df['label'] = harmful.map({True: 'Cyberbullying', False: 'Not Cyberbullying'})
                         dfs.append(df[['text', 'label']])
            except:
                pass
    
    if not dfs:
        return None
    
    full_df = pd.concat(dfs, ignore_index=True).dropna().sample(n=10000, random_state=42).reset_index(drop=True)
    # Limit size for speed of experiment if needed, but full data is better
    return full_df

def main():
    print("Loading data...")
    df = load_data()
    if df is None:
        print("No data found!")
        return

    print(f"Data loaded: {len(df)} samples")
    
    # Split once to ensure fair comparison
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    
    # Config 1: Baseline (App2.py default) -> Remove Stopwords=True, Lemmatize=True
    print("\n--- Running Baseline (Remove Stopwords=True, Lemmatize=True) ---")
    baseline = ExperimentClassifier(use_stopwords=True, use_lemmatization=True)
    res_base = baseline.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Config 2: Experimental -> Remove Stopwords=False, Lemmatize=False
    print("\n--- Running Experimental (Remove Stopwords=False, Lemmatize=False) ---")
    experiment = ExperimentClassifier(use_stopwords=False, use_lemmatization=False)
    res_exp = experiment.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("\n\n====== RESULTS COMPARISON ======")
    print(f"{'Metric':<25} | {'Baseline':<10} | {'Experimental':<10} | {'Diff'}")
    print("-" * 60)
    
    acc_base = res_base['accuracy']
    acc_exp = res_exp['accuracy']
    print(f"{'Accuracy':<25} | {acc_base:.4f}     | {acc_exp:.4f}     | {acc_exp - acc_base:+.4f}")
    
    f1_cb_base = res_base['report']['Cyberbullying']['f1-score']
    f1_cb_exp = res_exp['report']['Cyberbullying']['f1-score']
    print(f"{'F1 (Cyberbullying)':<25} | {f1_cb_base:.4f}     | {f1_cb_exp:.4f}     | {f1_cb_exp - f1_cb_base:+.4f}")

    rec_cb_base = res_base['report']['Cyberbullying']['recall']
    rec_cb_exp = res_exp['report']['Cyberbullying']['recall']
    print(f"{'Recall (Cyberbullying)':<25} | {rec_cb_base:.4f}     | {rec_cb_exp:.4f}     | {rec_cb_exp - rec_cb_base:+.4f}")
    
    print("\nBaseline Info:")
    print("Stopwords Removed: YES")
    print("Lemmatization: YES")
    
    print("\nExperimental Info:")
    print("Stopwords Removed: NO")
    print("Lemmatization: NO")

if __name__ == "__main__":
    main()

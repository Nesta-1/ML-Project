
import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

def load_data(filepath):
    """Loads the dataset from the given filepath."""
    df = pd.read_csv(filepath)
    return df

def clean_text(text):
    """
    Applies text cleaning steps:
    - Lowercase
    - Remove URLs
    - Remove special characters (keep only alphanumeric and spaces)
    - Demojize (convert emojis to text)
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Handle emojis
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Remove special characters and numbers (keep letters and spaces)
    # The paper mentions removing special characters.
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Full preprocessing pipeline:
    - Clean text
    - Tokenize (split by space)
    - Remove Stopwords
    - Lemmatize
    """
    text = clean_text(text)
    words = text.split()
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
        
    return " ".join(words)

def apply_sentiment_filtering(df, text_col='tweet_text'):
    """
    Filters out tweets with non-negative sentiment (polarity >= 0)
    as per the paper's methodology.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    def get_polarity(text):
        return analyzer.polarity_scores(text)['compound']
    
    # Calculate polarity
    df['polarity'] = df[text_col].apply(get_polarity)
    
    # Filter: Keep only negative sentiment (polarity < 0)
    # The paper says: "remove any tweets that have a polarity score of zero or above"
    # So we keep tweets with polarity < 0.
    filtered_df = df[df['polarity'] < 0].copy()
    
    return filtered_df

def prepare_data(filepath, text_col='tweet_text', label_col='cyberbullying_type', 
                 use_sentiment_filter=False, apply_smote=False):
    """
    Orchestrates the data preparation process.
    """
    print("Loading data...")
    df = load_data(filepath)
    
    print("Preprocessing text...")
    df['clean_text'] = df[text_col].apply(preprocess_text)
    
    if use_sentiment_filter:
        print("Applying sentiment filtering...")
        initial_count = len(df)
        df = apply_sentiment_filtering(df, text_col='clean_text')
        print(f"Filtered out {initial_count - len(df)} rows. Remaining: {len(df)}")
        
    # Split data
    X = df['clean_text']
    y = df[label_col]
    
    # Label encoding is needed for some models, but sklearn handles strings for y often.
    # However, XGBoost needs numeric y.
    # Let's map classes to integers.
    unique_classes = y.unique()
    class_map = {label: idx for idx, label in enumerate(unique_classes)}
    y = y.map(class_map)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features for memory/speed
    X_vec = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42, stratify=y)
    
    if apply_smote:
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    return X_train, X_test, y_train, y_test, vectorizer, class_map

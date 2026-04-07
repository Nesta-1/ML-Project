import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import lightgbm as lgb
import scipy.sparse as sp
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import joblib
import emoji
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import json

# Page configuration — must be first Streamlit command
st.set_page_config(
    page_title="CyberGuard AI - Cyberbullying Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NLTK data download (optional; app works offline with fallback tokenization)
@st.cache_resource
def download_nltk_data():
    import sys
    import io
    required_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for item in required_downloads:
        try:
            # Suppress NLTK's stderr (e.g. "getaddrinfo failed" when offline)
            _old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                nltk.download(item, quiet=True)
            finally:
                sys.stderr = _old_stderr
        except Exception:
            pass

download_nltk_data()

# Initialize components
analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Animated gradient UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    @keyframes bgShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.4; }
        33% { transform: translate(30px, -20px) scale(1.05); opacity: 0.6; }
        66% { transform: translate(-20px, 20px) scale(0.95); opacity: 0.5; }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.2); }
        50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.35); }
    }
    
    .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        background: linear-gradient(-45deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f0f23 75%, #1a1a2e 100%);
        background-size: 400% 400%;
        animation: bgShift 15s ease infinite;
        color: #e2e8f0;
        min-height: 100vh;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 20% 30%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 70%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                    radial-gradient(ellipse at 50% 50%, rgba(59, 130, 246, 0.06) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
        animation: float 20s ease-in-out infinite;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 720px;
        margin-left: auto;
        margin-right: auto;
        position: relative;
        z-index: 1;
    }
    
    .app-hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 0;
    }
    .app-hero .hero-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        filter: drop-shadow(0 0 12px rgba(99, 102, 241, 0.4));
    }
    .app-hero .hero-title {
        color: #f1f5f9;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.75rem 0;
        display: block;
        line-height: 1.3;
        text-shadow: 0 0 20px rgba(165, 180, 252, 0.3);
    }
    .app-hero .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }
    .input-section {
        margin-top: 0.75rem;
        padding: 0.5rem 0;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.25);
        position: relative;
        z-index: 1;
    }
    
    .header-container {
        padding: 1.25rem 1.5rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.15) 100%);
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        animation: glow 4s ease-in-out infinite;
    }
    
    .title-text {
        background: linear-gradient(135deg, #a5b4fc 0%, #c4b5fd 50%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .prediction-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        padding: 1.5rem 1.75rem;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.25);
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .vader-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        padding: 1.5rem 1.75rem;
        border-radius: 16px;
        border: 1px solid rgba(139, 92, 246, 0.25);
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .status-safe {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    
    .status-danger {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    }
    
    .metric-container { margin: 1.25rem 0; }
    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .progress-container {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 10px;
        height: 10px;
        margin: 0.75rem 0;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .progress-bar {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 6px;
        transition: width 0.4s ease;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
    }
    .progress-text {
        color: white;
        font-weight: 600;
        font-size: 0.7rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    .vader-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }
    .vader-metric:last-child { border-bottom: none; }
    .vader-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .vader-value {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.95rem;
        font-family: ui-monospace, monospace;
    }
    
    .tokens-container {
        background: rgba(15, 23, 42, 0.6);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .token-badge {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.35) 0%, rgba(139, 92, 246, 0.35) 100%);
        color: #e2e8f0;
        padding: 0.4rem 0.85rem;
        border-radius: 8px;
        font-size: 0.8rem;
        margin: 0.3rem 0.3rem 0.3rem 0;
        display: inline-block;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .section-header {
        color: #f1f5f9;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .stTextArea textarea {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 0.9rem 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    
    .sidebar-header {
        padding: 1rem 0;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.25);
    }
    .sidebar-title {
        background: linear-gradient(135deg, #a5b4fc 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .sidebar-subtitle {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    .status-indicator {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        margin-top: 1rem;
    }
    .status-text { color: #a5b4fc; font-size: 0.8rem; font-weight: 600; }
    
    .results-section {
        margin-top: 2.5rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(99, 102, 241, 0.2);
    }
    .results-title {
        text-align: center;
        margin-bottom: 1.75rem;
    }
    .results-title .results-heading {
        background: linear-gradient(135deg, #a5b4fc 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }
    .results-title .results-sub {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0;
    }
    .results-cards {
        margin-bottom: 2rem;
    }
    .results-cards .prediction-card,
    .results-cards .vader-card {
        margin: 0 0 1.5rem 0;
        padding: 1.75rem 2rem;
    }
    .results-tokens {
        margin-top: 2rem;
    }
    .results-tokens .section-header {
        margin-bottom: 0.75rem;
    }
    .results-tokens .tokens-container {
        margin-top: 0.75rem;
    }
    
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700; }
    .stMetric label { color: #94a3b8 !important; }
    .stAlert { border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.25); }
    
    p, li { color: #cbd5e1 !important; }
    .stMarkdown { color: #cbd5e1; }
    
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

from sklearn.preprocessing import LabelEncoder

def render_comparison_page():
    st.markdown("""
    <div class="app-hero">
        <span class="hero-logo">📈</span>
        <h1 class="hero-title">Model Performance Comparison</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "sga_net", "results", "comparison_results.json")
    
    if not os.path.exists(results_path):
        st.warning(f"Results file not found at {results_path}. Please run training evaluation first.")
        return

    try:
        with open(results_path, "r") as f:
            results = json.load(f)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return

    if not results:
        st.info("No results found in the file.")
        return

    # Create comparison table
    data = []
    for model_name, metrics in results.items():
        data.append({
            "Model": model_name.upper(),
            "Accuracy": metrics.get("accuracy", 0),
            "Macro F1": metrics.get("macro_f1", 0),
            "Test Loss": metrics.get("test_loss", 0)
        })
    
    if data:
        df = pd.DataFrame(data).set_index("Model")
        st.markdown("### 📊 Metrics Overview")
        st.table(df.style.format("{:.4f}").highlight_max(axis=0, subset=["Accuracy", "Macro F1"]))

        # Bar chart
        st.markdown("### 🏆 Performance Comparison")
        df_melted = df.reset_index().melt(id_vars="Model", value_vars=["Accuracy", "Macro F1"], var_name="Metric", value_name="Score")
        
        fig = px.bar(df_melted, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=["#6366f1", "#8b5cf6"],
                     title="Accuracy vs F1 Score")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    # Per-class F1 Heatmap
    st.markdown("### 🧬 Per-Class F1 Scores")
    
    class_data = {}
    for model_name, metrics in results.items():
        if "per_class_f1" in metrics:
            class_data[model_name.upper()] = metrics["per_class_f1"]
            
    if class_data:
        df_class = pd.DataFrame(class_data).T
        fig2 = px.imshow(df_class, text_auto=".2f", aspect="auto",
                         color_continuous_scale="Viridis",
                         title="F1 Score by Cyberbullying Type")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig2, use_container_width=True)


# Curated profanity / slur word list for profanity_score feature
_PROFANITY_WORDS = frozenset([
    'fuck', 'shit', 'bitch', 'ass', 'damn', 'hell', 'crap', 'dick', 'piss',
    'bastard', 'slut', 'whore', 'idiot', 'stupid', 'dumb', 'moron', 'retard',
    'ugly', 'fat', 'loser', 'trash', 'scum', 'pathetic', 'worthless', 'die',
    'kill', 'hate', 'stfu', 'wtf', 'fk', 'fck', 'sht', 'btch', 'negro',
    'nigger', 'faggot', 'fag', 'dyke', 'homo', 'terrorist', 'kys',
    'suicide', 'rape', 'molest', 'freak', 'creep', 'psycho', 'weirdo',
])


class CyberbullyingClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.scaler = None          # StandardScaler for numeric features
        self.is_trained = False
        self.training_data = None
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        try:
            text = text.lower()
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
                if token not in stop_words and len(token) > 0:
                    try:
                        lemmatized_tokens.append(lemmatizer.lemmatize(token))
                    except:
                        lemmatized_tokens.append(token)
            
            return ' '.join(lemmatized_tokens), lemmatized_tokens
        except:
            clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            tokens = [word for word in clean_text.split() if len(word) > 0]
            return ' '.join(tokens), tokens
    
    def get_vader_features(self, text):
        """Extract VADER sentiment features"""
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
    
    def extract_meta_features(self, text):
        """Extract 11 hand-crafted stylistic / structural features from raw text."""
        words = text.split()
        alpha_chars = [c for c in text if c.isalpha()]
        upper_chars = [c for c in alpha_chars if c.isupper()]
        return [
            len(text),                                                   # text_length
            len(words),                                                  # word_count
            np.mean([len(w) for w in words]) if words else 0.0,          # avg_word_length
            len(upper_chars) / max(len(alpha_chars), 1),                 # uppercase_ratio
            text.count('!'),                                             # exclamation_count
            text.count('?'),                                             # question_count
            sum(1 for w in words if w.startswith('#')),                   # hashtag_count
            sum(1 for w in words if w.startswith('@')),                   # mention_count
            emoji.emoji_count(text),                                     # emoji_count
            sum(1 for w in text.lower().split() if w in _PROFANITY_WORDS),# profanity_score
            len(re.findall(r'(.)\1{2,}', text)),                         # repeated_char_count
        ]
    
    def _build_numeric_features(self, raw_texts):
        """Build the numeric feature matrix (VADER + meta) for a list of raw texts."""
        rows = []
        for text in raw_texts:
            vader = self.get_vader_features(text)
            meta = self.extract_meta_features(text)
            rows.append([
                vader['compound'], vader['positive'], vader['neutral'], vader['negative'],
                *meta
            ])
        return np.array(rows, dtype=np.float64)
    
    def load_dataset(self, uploaded_file):
        """Load and process uploaded dataset"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Please upload a CSV or Excel file")
                return False
            
            # Assume columns are 'text' and 'label'
            if 'text' not in df.columns or 'label' not in df.columns:
                st.error("Dataset must have 'text' and 'label' columns")
                return False
            
            self.training_data = df
            st.success(f"Successfully loaded {len(df)} samples from dataset")
            return True
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return False
    
    def load_dataset_from_path(self, filepath):
        """Load CSV from path. Sets self.training_data. Supports classified_tweets.csv and cyberbullying_tweets.csv formats."""
        df = self._load_csv_to_dataframe(filepath)
        if df is not None:
            self.training_data = df
            return True
        return False
    
    def _load_csv_to_dataframe(self, filepath):
        """Load a CSV and return a DataFrame with columns text, label; or None if unsupported format."""
        try:
            df = pd.read_csv(filepath)
            # cyberbullying_tweets.csv: tweet_text, cyberbullying_type
            if 'tweet_text' in df.columns and 'cyberbullying_type' in df.columns:
                df = df.rename(columns={'tweet_text': 'text'})
                df['label'] = df['cyberbullying_type']
                return df[['text', 'label']].dropna(subset=['text'])
            # classified_tweets.csv: text, suspicious, cyberbullying, hate, suicidal
            if 'text' in df.columns and 'cyberbullying' in df.columns and 'hate' in df.columns and 'suicidal' in df.columns:
                # This dataset is binary, so we map it to 'other_cyberbullying' or 'not_cyberbullying'
                harmful = (df['cyberbullying'].fillna(0) >= 1) | (df['hate'].fillna(0) >= 1) | (df['suicidal'].fillna(0) >= 1)
                df = df.copy()
                df['label'] = harmful.map({True: 'other_cyberbullying', False: 'not_cyberbullying'})
                return df[['text', 'label']].dropna(subset=['text'])
            # already has text + label
            if 'text' in df.columns and 'label' in df.columns:
                return df[['text', 'label']].dropna(subset=['text'])
            return None
        except Exception:
            return None
    
    def train_model(self, use_dataset=False):
        """Train a tuned LightGBM classifier with rich features."""
        try:
            if use_dataset and self.training_data is not None:
                texts = self.training_data['text'].tolist()
                labels = self.training_data['label'].tolist()
            else:
                # Synthetic fallback data
                synthetic_data = [
                    ("you are a bitch", "gender"), ("sexist pig", "gender"),
                    ("go back to the kitchen", "gender"), ("gay", "gender"),
                    ("terrorist", "religion"), ("muslims are bad", "religion"),
                    ("jewish scum", "religion"),
                    ("white trash", "ethnicity"), ("black monkeys", "ethnicity"),
                    ("nigger", "ethnicity"),
                    ("kid", "age"), ("school shooter", "age"),
                    ("middle school loser", "age"),
                    ("you're such an idiot", "other_cyberbullying"),
                    ("kill yourself", "other_cyberbullying"),
                    ("you're ugly and stupid", "other_cyberbullying"),
                    ("shut up loser", "other_cyberbullying"),
                    ("fk you", "other_cyberbullying"),
                    ("fuck you", "other_cyberbullying"),
                    ("u r stupid", "other_cyberbullying"),
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
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(t)[0] for t in texts]
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(labels)
            num_classes = len(self.label_encoder.classes_)
            
            # ── Feature engineering ──────────────────────────────────────
            # 1) TF-IDF (5K features, sublinear TF, bigrams+trigrams)
            self.vectorizer = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 3),
                sublinear_tf=True, min_df=2, max_df=0.95
            )
            X_tfidf = self.vectorizer.fit_transform(processed_texts)
            
            # 2) Numeric features: VADER (4) + hand-crafted meta (11) = 15
            X_numeric = self._build_numeric_features(texts)
            
            # 3) Scale numeric features
            self.scaler = StandardScaler()
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            
            # 4) Combine: sparse TF-IDF + dense numeric
            X_combined = sp.hstack([X_tfidf, sp.csr_matrix(X_numeric_scaled)], format='csr')
            
            # ── Train / test split ───────────────────────────────────────
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # No oversampling needed — dataset is already balanced (~8K per class)
            
            # ── LightGBM (fast, high accuracy) ──────────────────────────
            self.model = lgb.LGBMClassifier(
                n_estimators=600, max_depth=12, learning_rate=0.08,
                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.05, reg_lambda=0.1, min_child_samples=20,
                random_state=42, verbose=-1, n_jobs=-1,
                num_class=num_classes, class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Calculate accuracy
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            return train_score, test_score
            
        except Exception as e:
            st.error(f"Training error: {e}")
            return None, None
    
    def predict(self, text):
        """Predict cyberbullying classification (Multi-class)"""
        try:
            if not self.is_trained:
                with st.spinner("Training model..."):
                    self.train_model()
            
            if not self.is_trained:
                return "Error", 0.0, {'compound': 0.0, 'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}, [], None
            
            # Preprocess text
            processed_text, tokens = self.preprocess_text(text)
            
            # Build features (same pipeline as training)
            X_tfidf = self.vectorizer.transform([processed_text])
            X_numeric = self._build_numeric_features([text])
            X_numeric_scaled = self.scaler.transform(X_numeric)
            X_combined = sp.hstack([X_tfidf, sp.csr_matrix(X_numeric_scaled)], format='csr')
            
            # Get VADER for display
            vader = self.get_vader_features(text)
            
            # Predict probabilities
            probabilities = self.model.predict_proba(X_combined)[0]
            
            # Get class labels from LabelEncoder if available, else from model classes
            if self.label_encoder:
                class_labels = self.label_encoder.classes_
            else:
                class_labels = self.model.classes_

            # Find "not_cyberbullying" index
            not_cb_index = -1
            for i, label in enumerate(class_labels):
                if label == 'not_cyberbullying' or label == 'Not Cyberbullying':
                    not_cb_index = i
                    break
            
            # Calculate Prob(Bullying) = 1 - Prob(Safe)
            if not_cb_index != -1:
                prob_safe = probabilities[not_cb_index]
                prob_bullying = 1.0 - prob_safe
            else:
                prob_bullying = np.max(probabilities)
                prob_safe = 1.0 - prob_bullying

            # Threshold Logic (0.60)
            THRESHOLD = 0.60
            specific_category = None
            
            if prob_bullying > THRESHOLD:
                prediction = "Cyberbullying"
                confidence = prob_bullying
                
                probs_copy = probabilities.copy()
                if not_cb_index != -1:
                    probs_copy[not_cb_index] = -1.0
                
                max_idx = np.argmax(probs_copy)
                specific_category = class_labels[max_idx]
                specific_category = specific_category.replace('_', ' ').title()
                if specific_category == "Not Cyberbullying":
                    specific_category = "General"
                    
            else:
                prediction = "Not Cyberbullying"
                confidence = prob_safe
            
            return prediction, confidence, vader, tokens, specific_category
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return "Error", 0.0, {'compound': 0.0, 'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}, [], None

# Initialize classifier: load from cache if available, else train and save
CACHE_FILENAME = 'cyberguard_model.joblib'
# Save in current working directory (folder where you run: streamlit run app2.py)
_cache_dir = os.getcwd()
CACHE_PATH = os.path.join(_cache_dir, CACHE_FILENAME)
# CSV files are next to the script
_script_dir = os.path.dirname(os.path.abspath(__file__))

if 'classifier' not in st.session_state:
    st.session_state.classifier = CyberbullyingClassifier()
    st.session_state.model_loaded_from_cache = False
    loaded = False
    if os.path.isfile(CACHE_PATH):
        try:
            data = joblib.load(CACHE_PATH)
            st.session_state.classifier.model = data['model']
            st.session_state.classifier.vectorizer = data['vectorizer']
            if 'label_encoder' in data:
                st.session_state.classifier.label_encoder = data['label_encoder']
            if 'scaler' in data:
                st.session_state.classifier.scaler = data['scaler']
            st.session_state.classifier.is_trained = True
            loaded = True
            st.session_state.model_loaded_from_cache = True
        except Exception:
            pass
    if not loaded:
        base_dir = _script_dir
        csv_files = ['cyberbullying_tweets.csv']
        dfs = []
        for name in csv_files:
            path = os.path.join(base_dir, name)
            if os.path.isfile(path):
                df = st.session_state.classifier._load_csv_to_dataframe(path)
                if df is not None:
                    dfs.append(df)
        if dfs:
            st.session_state.classifier.training_data = pd.concat(dfs, ignore_index=True)
            with st.spinner("Training LightGBM on cyberbullying_tweets.csv..."):
                st.session_state.classifier.train_model(use_dataset=True)
        else:
            st.session_state.classifier.train_model(use_dataset=False)
        if st.session_state.classifier.is_trained:
            try:
                joblib.dump({
                    'model': st.session_state.classifier.model,
                    'vectorizer': st.session_state.classifier.vectorizer,
                    'label_encoder': st.session_state.classifier.label_encoder,
                    'scaler': st.session_state.classifier.scaler,
                }, CACHE_PATH)
                st.session_state.model_cache_saved_path = CACHE_PATH
            except Exception as e:
                st.session_state.model_cache_saved_path = None
                st.session_state.model_cache_save_error = str(e)

# Sidebar — minimal
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">CyberGuard</div>
        <div class="sidebar-subtitle">Cyberbullying Classifier</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("Navigation", ["🛡️ Classifier", "📈 Model Comparison"], label_visibility="collapsed")

if page == "📈 Model Comparison":
    render_comparison_page()
    st.stop()

# Main content — single Test page, centered
st.markdown("""
<div class="app-hero">
    <span class="hero-logo">🛡️</span>
    <h1 class="hero-title">CyberGuard — Cyberbullying Classifier</h1>
</div>
<div class="input-section">
""", unsafe_allow_html=True)

input_text = st.text_area(
    "Message",
    value="Thank you for your help today!",
    height=140,
    key="input_text",
    placeholder="Paste or type text here..."
)

run_inference = st.button("Analyze", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if run_inference and input_text.strip():
    with st.spinner("Analyzing message..."):
        prediction, confidence, vader_scores, tokens, specific_category = st.session_state.classifier.predict(input_text)
    
    if prediction != "Error":
        st.markdown("""
        <div class="results-section">
            <div class="results-title">
                <p class="results-heading">CyberGuard — Cyberbullying Classifier</p>
                <p class="results-sub">Analysis result</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="results-cards">', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            is_cyberbullying = prediction == "Cyberbullying"
            if is_cyberbullying:
                status_class = "status-danger"
                if specific_category:
                    status_text = f"Harassment Detected ({specific_category})"
                else:
                    status_text = "Harassment Detected"
            else:
                status_class = "status-safe"
                status_text = "Safe"
            
            st.markdown(f"""
            <div class="prediction-card">
                <div class="{status_class}">{status_text}</div>
                <div class="metric-container">
                    <div class="metric-label">Detected Category</div>
                    <div class="metric-value">{prediction}</div>
                </div>
                <div class="metric-container">
                    <div class="metric-label">Model Confidence</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {confidence*100:.1f}%;">
                            <span class="progress-text">{confidence*100:.1f}%</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="vader-card">
                <div class="section-header">Sentiment (VADER)</div>
                <div class="vader-metric">
                    <span class="vader-label">Compound Score</span>
                    <span class="vader-value">{vader_scores['compound']:.3f}</span>
                </div>
                <div class="vader-metric">
                    <span class="vader-label">Positive</span>
                    <span class="vader-value">{vader_scores['positive']:.3f}</span>
                </div>
                <div class="vader-metric">
                    <span class="vader-label">Neutral</span>
                    <span class="vader-value">{vader_scores['neutral']:.3f}</span>
                </div>
                <div class="vader-metric">
                    <span class="vader-label">Negative</span>
                    <span class="vader-value">{vader_scores['negative']:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="results-tokens">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Tokens</div>', unsafe_allow_html=True)
        if tokens:
            tokens_html = ''.join([f'<span class="token-badge">{token}</span>' for token in tokens[:20]])
            st.markdown(f'<div class="tokens-container">{tokens_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tokens-container"><span style="color: #94a3b8;">No tokens after preprocessing</span></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

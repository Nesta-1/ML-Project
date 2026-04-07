import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
import re
import os
import emoji

# Ensure the page config is set
st.set_page_config(
    page_title="CyberGuard AI - Cyberbullying Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dynamically load the exact UI CSS from app2.py
@st.cache_data
def load_css():
    try:
        with open("app2.py", "r", encoding="utf-8") as f:
            content = f.read()
        css_match = re.search(r'st\.markdown\("""\n*(<style>.*?</style>)\n*""", unsafe_allow_html=True\)', content, re.DOTALL)
        if css_match:
            return css_match.group(1)
    except Exception:
        pass
    return ""

css = load_css()
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load the backend (ml2.py)
# ---------------------------------------------------------
@st.cache_resource
def load_ml2_model():
    with st.spinner("Initializing Model Fast (Skipping dataset rebuild)..."):
        import types
        import joblib
        
        with open("ml2.py", "r", encoding="utf-8") as f:
            code = f.read()
        
        # 1. Prevent building features for the 47,000 strings
        cutoff_marker = "# 7) BUILD DATASET"
        if cutoff_marker in code:
            code = code[:code.find(cutoff_marker)]
            
        # 2. Prevent loading the massive CSV to memory just to get the module definitions
        code = code.replace(
            'df=pd.read_csv("cyberbullying_tweets.csv")',
            'df=pd.DataFrame(columns=["tweet_text", "cyberbullying_type"])'
        )
        
        # Execute the module safely in its own namespace
        ml2_mod = types.ModuleType("ml2")
        ml2_mod.__file__ = "ml2.py"
        exec(code, ml2_mod.__dict__)
        
        # Hook up the saved model caches to bypass training completely
        import __main__
        __main__.ClassAwareLexiconFeatures = ml2_mod.ClassAwareLexiconFeatures
        __main__.ClassCentroidSimilarity = ml2_mod.ClassCentroidSimilarity
        
        ml2_mod.pipeline = joblib.load("ml2_pipeline.joblib")
        ml2_mod.label_encoder = joblib.load("ml2_label_encoder.joblib")
        
        return ml2_mod

ml2 = load_ml2_model()
analyzer = SentimentIntensityAnalyzer()

def predict(text):
    # Construct exactly the dataframe features that ml2 expects
    df_single = pd.DataFrame({"tweet_text": [text]})
    df_single["clean_text"] = df_single["tweet_text"].apply(ml2.normalize_text)
    df_single["masked_text"] = df_single["tweet_text"].apply(ml2.mask_structural)
    df_single["interaction_text"] = df_single["tweet_text"].apply(ml2.interaction_tokens)
    
    num_df_series = df_single["tweet_text"].apply(ml2.build_numeric_features)
    
    if isinstance(num_df_series, pd.Series):
        if len(num_df_series) > 0 and isinstance(num_df_series.iloc[0], pd.Series):
            num_df = num_df_series.apply(pd.Series)
        else:
            num_df = pd.DataFrame(num_df_series).T
    else:
        num_df = num_df_series

    X_single = pd.concat([df_single, num_df], axis=1)
    
    # Run probabilities through ml2
    probabilities = ml2.pipeline.predict_proba(X_single)[0]
    class_labels = ml2.label_encoder.classes_
    
    not_cb_index = -1
    for i, label in enumerate(class_labels):
        if label == 'not_cyberbullying' or label == 'Not Cyberbullying':
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
        confidence = prob_bullying
        
        probs_copy = probabilities.copy()
        if not_cb_index != -1:
            probs_copy[not_cb_index] = -1.0
        
        max_idx = np.argmax(probs_copy)
        specific_category = class_labels[max_idx]
        specific_category = str(specific_category).replace('_', ' ').title()
        if specific_category == "Not Cyberbullying":
            specific_category = "General"
    else:
        prediction = "Not Cyberbullying"
        confidence = prob_safe
        
    vader = analyzer.polarity_scores(text)
    vader_features = {
        'compound': vader['compound'],
        'positive': vader['pos'],
        'neutral': vader['neu'],
        'negative': vader['neg']
    }
    
    clean_text = ml2.normalize_text(text)
    tokens = [w for w in clean_text.split() if w]
    
    return prediction, confidence, vader_features, tokens, specific_category

# ---------------------------------------------------------
# Exact Frontend Layout from app2.py
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">CyberGuard</div>
        <div class="sidebar-subtitle">ML2 Classifier Base</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="app-hero">
    <span class="hero-logo">🛡️</span>
    <h1 class="hero-title">CyberGuard — ML2 Backend</h1>
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
        prediction, confidence, vader_scores, tokens, specific_category = predict(input_text)
    
    if prediction != "Error":
        st.markdown("""
        <div class="results-section">
            <div class="results-title">
                <p class="results-heading">Analysis result</p>
                <p class="results-sub">ML2 Soft Prototype Model</p>
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
                <div class="section-header">Sentiment Info (VADER)</div>
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

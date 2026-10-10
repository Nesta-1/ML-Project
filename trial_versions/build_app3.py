import os, re

try:
    with open("app2.py", "r", encoding="utf-8") as f:
        app2 = f.read()

    css = re.search(r'(st\.markdown\("""\n<style>.*?</style>\n""", unsafe_allow_html=True\))', app2, re.DOTALL).group(1)
    ui_hero = re.search(r'(# Main content — single Test page, centered.*?st\.markdown\("</div>", unsafe_allow_html=True\))', app2, re.DOTALL).group(1)
    ui_results = re.search(r'(if run_inference and input_text\.strip\(\):.*?st\.markdown\(\'</div></div>\', unsafe_allow_html=True\))', app2, re.DOTALL).group(1)

    with open("app_ml2.py", "w", encoding="utf-8") as f:
        f.write('''import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
import re
import os
import emoji

st.set_page_config(
    page_title="CyberGuard AI - Cyberbullying Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

''')
        f.write(css + "\n\n")
        f.write('''@st.cache_resource
def load_ml2_model():
    with st.spinner("Loading and Training ML2 Model... This will take a while for the first load."):
        import ml2
        return ml2

ml2 = load_ml2_model()

analyzer = SentimentIntensityAnalyzer()

def predict(text):
    df_single = pd.DataFrame({"tweet_text": [text]})
    df_single["clean_text"] = df_single["tweet_text"].apply(ml2.normalize_text)
    df_single["masked_text"] = df_single["tweet_text"].apply(ml2.mask_structural)
    df_single["interaction_text"] = df_single["tweet_text"].apply(ml2.interaction_tokens)
    num_df = df_single["tweet_text"].apply(ml2.build_numeric_features)
    
    if isinstance(num_df, pd.Series):
        if len(num_df) > 0 and isinstance(num_df.iloc[0], pd.Series):
            num_df = num_df.apply(pd.Series)
        else:
            num_df = pd.DataFrame(num_df).T

    X_single = pd.concat([df_single, num_df], axis=1)
    
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
        specific_category = specific_category.replace('_', ' ').title()
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

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">CyberGuard</div>
        <div class="sidebar-subtitle">ML2 Classifier Core</div>
    </div>
    """, unsafe_allow_html=True)
    
''')
        f.write(ui_hero + "\n\n")
        f.write(ui_results.replace('st.session_state.classifier.predict(input_text)', 'predict(input_text)') + "\n")
        
    print("app_ml2.py built successfully.")
except Exception as e:
    print(f"Error building app_ml2.py: {e}")

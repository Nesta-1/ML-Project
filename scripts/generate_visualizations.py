import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud
from collections import Counter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import numpy as np

# Ensure visualizations directory exists
os.makedirs('visualizations', exist_ok=True)

# Load datasets
def load_data():
    cyber_df = pd.read_csv('cyberbullying_tweets.csv')
    try:
        classified_df = pd.read_csv('classified_tweets.csv')
    except:
        classified_df = None
    return cyber_df, classified_df

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    return text

print("Loading data...")
cyber_df, classified_df = load_data()
cyber_df['clean_text'] = cyber_df['tweet_text'].apply(preprocess_text)

print("Generating visualizations...")

# 1. Class Distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='cyberbullying_type', data=cyber_df, order=cyber_df['cyberbullying_type'].value_counts().index)
plt.title('Distribution of Cyberbullying Classes')
plt.xlabel('Count')
plt.ylabel('Class')
plt.tight_layout()
plt.savefig('visualizations/01_class_distribution.png')
plt.close()

# 2. Text Length Distribution
cyber_df['text_length'] = cyber_df['tweet_text'].astype(str).apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(cyber_df['text_length'], bins=50, kde=True)
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Length')
plt.savefig('visualizations/02_text_length_distribution.png')
plt.close()

# 3. Word Count Distribution
cyber_df['word_count'] = cyber_df['tweet_text'].astype(str).apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(cyber_df['word_count'], bins=50, kde=True, color='orange')
plt.title('Distribution of Word Counts')
plt.xlabel('Word Count')
plt.savefig('visualizations/03_word_count_distribution.png')
plt.close()

# 4. Average Word Length by Class
cyber_df['avg_word_curr'] = cyber_df['tweet_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0)
plt.figure(figsize=(10, 6))
sns.boxplot(x='cyberbullying_type', y='avg_word_curr', data=cyber_df)
plt.title('Average Word Length by Class')
plt.xticks(rotation=45)
plt.savefig('visualizations/04_avg_word_length.png')
plt.close()

# 5. Sentiment Score Distribution
analyzer = SentimentIntensityAnalyzer()
cyber_df['sentiment'] = cyber_df['tweet_text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
plt.figure(figsize=(10, 6))
sns.histplot(cyber_df['sentiment'], bins=30, kde=True, color='green')
plt.title('Sentiment Score Distribution (VADER)')
plt.xlabel('Compound Score')
plt.savefig('visualizations/05_sentiment_distribution.png')
plt.close()

# 6. Sentiment by Class
plt.figure(figsize=(12, 6))
sns.boxplot(x='cyberbullying_type', y='sentiment', data=cyber_df)
plt.title('Sentiment Scores by Class')
plt.xticks(rotation=45)
plt.savefig('visualizations/06_sentiment_by_class.png')
plt.close()

# Helper for word stats
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def get_top_words(texts, n=20):
    all_words = ' '.join([str(t) for t in texts]).split()
    clean_words = [w for w in all_words if w not in stop_words and len(w) > 2]
    return Counter(clean_words).most_common(n)

# 7. Top 20 Words - Cyberbullying
bullying_texts = cyber_df[cyber_df['cyberbullying_type'] != 'not_cyberbullying']['clean_text']
top_bullying = get_top_words(bullying_texts)
words, counts = zip(*top_bullying)
plt.figure(figsize=(10, 8))
sns.barplot(x=list(counts), y=list(words))
plt.title('Top 20 Words in Cyberbullying Tweets')
plt.savefig('visualizations/07_top_words_bullying.png')
plt.close()

# 8. Top 20 Words - Non-Cyberbullying
safe_texts = cyber_df[cyber_df['cyberbullying_type'] == 'not_cyberbullying']['clean_text']
top_safe = get_top_words(safe_texts)
words, counts = zip(*top_safe)
plt.figure(figsize=(10, 8))
sns.barplot(x=list(counts), y=list(words))
plt.title('Top 20 Words in Non-Cyberbullying Tweets')
plt.savefig('visualizations/08_top_words_safe.png')
plt.close()

# 9. Word Cloud - Cyberbullying
wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(bullying_texts))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Cyberbullying')
plt.savefig('visualizations/09_wordcloud_bullying.png')
plt.close()

# 10. Word Cloud - Non-Cyberbullying
wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(safe_texts))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Safe Tweets')
plt.savefig('visualizations/10_wordcloud_safe.png')
plt.close()

# 11. Bigrams
def get_top_bigrams(texts, n=15):
    all_words = ' '.join([str(t) for t in texts]).split()
    clean_words = [w for w in all_words if w not in stop_words and len(w) > 2]
    bigrams = zip(clean_words, clean_words[1:])
    return Counter(bigrams).most_common(n)

top_bi = get_top_bigrams(bullying_texts)
labels, values = zip(*top_bi)
labels = [' '.join(l) for l in labels]
plt.figure(figsize=(10, 8))
sns.barplot(x=list(values), y=list(labels))
plt.title('Top Bigrams in Cyberbullying')
plt.savefig('visualizations/11_bigrams_bullying.png')
plt.close()

# 12. Special Characters
def count_special(text):
    return sum([1 for char in str(text) if char in ['!', '?', '*', '$', '#', '@']])

cyber_df['special_chars'] = cyber_df['tweet_text'].apply(count_special)
plt.figure(figsize=(10, 6))
sns.barplot(x='cyberbullying_type', y='special_chars', data=cyber_df)
plt.title('Average Special Character Usage by Class')
plt.xticks(rotation=45)
plt.savefig('visualizations/12_special_chars_by_class.png')
plt.close()

# 13. Unique vs Total Words (Vocabulary Richness)
cyber_df['unique_words'] = cyber_df['tweet_text'].apply(lambda x: len(set(str(x).split())))
plt.figure(figsize=(10, 6))
sns.scatterplot(x='word_count', y='unique_words', hue='cyberbullying_type', data=cyber_df, alpha=0.5)
plt.title('Vocabulary Richness (Unique vs Total Words)')
plt.savefig('visualizations/13_vocab_richness.png')
plt.close()

# 14. Classified Tweets Overlap (if available)
if classified_df is not None:
    # Assuming columns 'suspicious', 'cyberbullying', 'hate', 'suicidal' are binary or counts
    categories = ['suspicious', 'cyberbullying', 'hate', 'suicidal']
    sums = []
    for cat in categories:
        if cat in classified_df.columns:
            sums.append(classified_df[cat].sum())
        else:
            sums.append(0)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sums, labels=categories, autopct='%1.1f%%')
    plt.title('Distribution in Classified Tweets Dataset')
    plt.savefig('visualizations/14_classified_distribution.png')
    plt.close()
else:
    # Placeholder if file missing
    plt.figure()
    plt.text(0.5, 0.5, 'Classified Tweets Dataset Not Found', ha='center')
    plt.savefig('visualizations/14_classified_missing.png')
    plt.close()

# 15. Correlation Heatmap (Numeric features)
numeric_df = cyber_df[['text_length', 'word_count', 'sentiment', 'special_chars', 'unique_words']]
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation of Tweet Features')
plt.savefig('visualizations/15_feature_correlation.png')
plt.close()

print("All visualizations generated in 'visualizations/' folder.")

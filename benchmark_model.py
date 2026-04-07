"""
Benchmark: Old (ExtraTrees) vs New (LightGBM + rich features)
Run:  python benchmark_model.py
"""
import time, os, json, gc
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import lightgbm as lgb
import scipy.sparse as sp
import nltk, re, emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── helpers ──────────────────────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

_PROFANITY_WORDS = frozenset([
    'fuck', 'shit', 'bitch', 'ass', 'damn', 'hell', 'crap', 'dick', 'piss',
    'bastard', 'slut', 'whore', 'idiot', 'stupid', 'dumb', 'moron', 'retard',
    'ugly', 'fat', 'loser', 'trash', 'scum', 'pathetic', 'worthless', 'die',
    'kill', 'hate', 'stfu', 'wtf', 'fk', 'fck', 'sht', 'btch', 'negro',
    'nigger', 'faggot', 'fag', 'dyke', 'homo', 'terrorist', 'kys',
    'suicide', 'rape', 'molest', 'freak', 'creep', 'psycho', 'weirdo',
])


def preprocess(text):
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by'}
    out = []
    for t in tokens:
        if t not in stop_words and len(t) > 0:
            try:
                out.append(lemmatizer.lemmatize(t))
            except Exception:
                out.append(t)
    return ' '.join(out)


def vader_features(text):
    s = analyzer.polarity_scores(text)
    return [s['compound'], s['pos'], s['neu'], s['neg']]


def meta_features(text):
    words = text.split()
    alpha = [c for c in text if c.isalpha()]
    upper = [c for c in alpha if c.isupper()]
    return [
        len(text),
        len(words),
        np.mean([len(w) for w in words]) if words else 0.0,
        len(upper) / max(len(alpha), 1),
        text.count('!'),
        text.count('?'),
        sum(1 for w in words if w.startswith('#')),
        sum(1 for w in words if w.startswith('@')),
        emoji.emoji_count(text),
        sum(1 for w in text.lower().split() if w in _PROFANITY_WORDS),
        len(re.findall(r'(.)\1{2,}', text)),
    ]


# ── Data loading (only cyberbullying_tweets.csv) ─────────────────────────────
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(base, 'cyberbullying_tweets.csv')
    if os.path.isfile(p1):
        df = pd.read_csv(p1)
        df = df.rename(columns={'tweet_text': 'text'})
        df['label'] = df['cyberbullying_type']
        return df[['text', 'label']].dropna(subset=['text'])
    return None


# ── Build features (ALL SPARSE) ────────────────────────────────────────────
def build_old_features(processed, raw):
    """Old pipeline: TF-IDF 1500 + 4 VADER → sparse"""
    vec = TfidfVectorizer(max_features=1500, ngram_range=(1, 3))
    X_tfidf = vec.fit_transform(processed)
    V = np.array([vader_features(t) for t in raw])
    X = sp.hstack([X_tfidf, sp.csr_matrix(V)], format='csr')
    return X, vec


def build_new_features(processed, raw):
    """New pipeline: TF-IDF 5K (sublinear) + VADER(4) + meta(11) scaled → sparse"""
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), sublinear_tf=True, min_df=2, max_df=0.95)
    X_tfidf = vec.fit_transform(processed)
    numeric = np.array([vader_features(t) + meta_features(t) for t in raw], dtype=np.float64)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)
    X = sp.hstack([X_tfidf, sp.csr_matrix(numeric_scaled)], format='csr')
    return X, vec, scaler


# ── Models ───────────────────────────────────────────────────────────────────
def old_model():
    return ExtraTreesClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)


def new_model(num_classes):
    return lgb.LGBMClassifier(
        n_estimators=600, max_depth=12, learning_rate=0.08,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.05, reg_lambda=0.1, min_child_samples=20,
        random_state=42, verbose=-1, n_jobs=-1,
        num_class=num_classes, class_weight='balanced')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  BENCHMARK: Old (ExtraTrees) vs New (LightGBM + Rich Features)")
    print("=" * 70)

    df = load_data()
    if df is None:
        print("ERROR: cyberbullying_tweets.csv not found.")
        return
    print(f"\nDataset: {len(df)} samples, {df['label'].nunique()} classes")
    print(df['label'].value_counts().to_string())

    texts = df['text'].tolist()
    labels = df['label'].tolist()
    print("\nPreprocessing texts...")
    processed = [preprocess(t) for t in texts]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # ── Old features ──
    print("\n─── Building OLD features (TF-IDF 1500 + VADER 4) ───")
    t0 = time.time()
    X_old, _ = build_old_features(processed, texts)
    print(f"  Feature matrix: {X_old.shape}  ({time.time()-t0:.1f}s)")
    X_old_tr, X_old_te, y_tr, y_te = train_test_split(X_old, y, test_size=0.2, random_state=42, stratify=y)

    # ── Train OLD model ──
    print("\n─── Training OLD model (ExtraTreesClassifier) ───")
    t0 = time.time()
    m_old = old_model()
    m_old.fit(X_old_tr, y_tr)
    dur_old = time.time() - t0
    y_pred_old = m_old.predict(X_old_te)
    acc_old = accuracy_score(y_te, y_pred_old)
    f1_old = f1_score(y_te, y_pred_old, average='macro')
    print(f"  Train time: {dur_old:.1f}s")
    print(f"  Test Accuracy:  {acc_old:.4f}")
    print(f"  Test Macro-F1:  {f1_old:.4f}")
    print("\n  Classification Report (OLD):")
    print(classification_report(y_te, y_pred_old, target_names=le.classes_))

    results = {}
    results['old_extratrees'] = {
        'accuracy': round(acc_old, 4),
        'macro_f1': round(f1_old, 4),
        'train_time_s': round(dur_old, 1),
        'per_class_f1': {c: round(v, 4) for c, v in zip(le.classes_,
                         f1_score(y_te, y_pred_old, average=None))}
    }

    # Free old data
    del X_old, X_old_tr, m_old
    gc.collect()

    # ── New features ──
    print("\n─── Building NEW features (TF-IDF 5K + VADER 4 + Meta 11, scaled) ───")
    t0 = time.time()
    X_new, _, _ = build_new_features(processed, texts)
    print(f"  Feature matrix: {X_new.shape}  ({time.time()-t0:.1f}s)")
    X_new_tr, X_new_te, _, _ = train_test_split(X_new, y, test_size=0.2, random_state=42, stratify=y)

    # ── Train NEW model ──
    print("\n─── Training NEW model (LightGBM) ───")
    t0 = time.time()
    m_new = new_model(num_classes)
    m_new.fit(X_new_tr, y_tr)
    dur_new = time.time() - t0
    y_pred_new = m_new.predict(X_new_te)
    acc_new = accuracy_score(y_te, y_pred_new)
    f1_new = f1_score(y_te, y_pred_new, average='macro')
    print(f"  Train time: {dur_new:.1f}s")
    print(f"  Test Accuracy:  {acc_new:.4f}")
    print(f"  Test Macro-F1:  {f1_new:.4f}")
    print("\n  Classification Report (NEW):")
    print(classification_report(y_te, y_pred_new, target_names=le.classes_))
    results['new_lightgbm'] = {
        'accuracy': round(acc_new, 4),
        'macro_f1': round(f1_new, 4),
        'train_time_s': round(dur_new, 1),
        'per_class_f1': {c: round(v, 4) for c, v in zip(le.classes_,
                         f1_score(y_te, y_pred_new, average=None))}
    }

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    delta_acc = acc_new - acc_old
    delta_f1 = f1_new - f1_old
    print(f"  Old Accuracy: {acc_old:.4f}   |   New Accuracy: {acc_new:.4f}   |   Δ = {delta_acc:+.4f}")
    print(f"  Old Macro-F1: {f1_old:.4f}   |   New Macro-F1: {f1_new:.4f}   |   Δ = {delta_f1:+.4f}")
    print(f"  Old train time: {dur_old:.1f}s   |   New train time: {dur_new:.1f}s")
    print("=" * 70)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

import re
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
analyzer = SentimentIntensityAnalyzer()
df=pd.read_csv("cyberbullying_tweets.csv")
# ============================================================
# 1) DATA CHECKS + CLEANING
# ============================================================
required_cols = {"tweet_text", "cyberbullying_type"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_clean = df.drop_duplicates(subset=["tweet_text"]).copy()
df_clean = df_clean[df_clean["cyberbullying_type"] != "other_cyberbullying"].copy()
df_clean = df_clean.reset_index(drop=True)

# ============================================================
# 2) LEXICONS
# ============================================================
TARGET_TERMS = {
    "you", "u", "ur", "your", "yours", "yourself", "yall", "ya",
    "they", "them", "their", "he", "she", "him", "her"
}

SELF_TERMS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
}

INSULT_TERMS = {
    "idiot", "stupid", "dumb", "loser", "bitch", "bastard", "trash", "ugly",
    "hate", "pathetic", "fool", "shut", "annoying", "moron", "asshole",
    "jerk", "freak", "worthless", "disgusting", "clown", "brat", "suck"
}

NEGATIVE_WORDS = {
    "stupid", "idiot", "dumb", "loser", "trash", "ugly", "hate",
    "pathetic", "fool", "moron", "asshole", "jerk", "disgusting",
    "worthless", "suck", "bitch", "bastard", "annoying", "shut"
}

POSITIVE_WORDS = {
    "good", "great", "nice", "beautiful", "smart", "cool",
    "amazing", "kind", "love", "respect", "sweet", "pretty"
}

AGE_TERMS = {
    "age", "aged", "old", "older", "young", "younger", "kid", "kids", "child",
    "children", "teen", "teenager", "teenagers", "adult", "adults", "boy", "boys",
    "girl", "girls", "baby", "babies", "grandpa", "grandma", "grandmother", "grandfather"
}

GENDER_TERMS = {
    "gender", "boy", "boys", "girl", "girls", "man", "men", "woman", "women",
    "male", "female", "lady", "ladies", "guy", "guys", "he", "she", "him", "her",
    "his", "hers", "trans", "transgender"
}

ETHNICITY_TERMS = {
    "black", "white", "asian", "african", "european", "indian", "arab", "latino",
    "latin", "hispanic", "chinese", "japanese", "korean", "mexican", "turkish",
    "pakistani", "nigerian", "race", "racial", "ethnic"
}

RELIGION_TERMS = {
    "religion", "religious", "muslim", "islam", "christian", "christianity",
    "hindu", "hinduism", "jew", "jewish", "buddhist", "buddhism", "sikh",
    "sikhism", "atheist", "church", "mosque", "temple", "bible", "quran",
    "prayer", "pray", "god", "allah"
}

DEMO_LEXICONS = {
    "age": AGE_TERMS,
    "gender": GENDER_TERMS,
    "ethnicity": ETHNICITY_TERMS,
    "religion": RELIGION_TERMS,
}

OBFUSCATION_MAP = str.maketrans({
    "@": "a",
    "0": "o",
    "1": "i",
    "!": "i",
    "|": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "$": "s",
})

URL_RE = re.compile(r"http\S+|www\S+")
MENTION_RE = re.compile(r"@\w+")
PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")
ELONG_RE = re.compile(r"(.)\1{2,}")

# ============================================================
# 3) TEXT NORMALIZATION
# ============================================================
def normalize_text(text):
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = text.translate(OBFUSCATION_MAP)
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

def tokenize(text):
    return normalize_text(text).split()

def mask_structural(text):
    tokens = tokenize(text)
    out = []
    for tok in tokens:
        if tok in TARGET_TERMS:
            out.append("TGT")
        elif tok in SELF_TERMS:
            out.append("SLF")
        else:
            out.append(tok)
    return " ".join(out)

def interaction_tokens(text):
    tokens = tokenize(text)
    markers = []

    tgt = any(t in TARGET_TERMS for t in tokens)
    slf = any(t in SELF_TERMS for t in tokens)
    neg = any(t in NEGATIVE_WORDS for t in tokens)
    pos = any(t in POSITIVE_WORDS for t in tokens)
    insult_hits = [t for t in tokens if t in INSULT_TERMS]

    if tgt and neg:
        markers.append("TGT_NEGATIVE")
    if tgt and pos:
        markers.append("TGT_POSITIVE")
    if slf and neg:
        markers.append("SLF_NEGATIVE")
    if tgt and insult_hits:
        markers.append("TGT_INSULT")
    if slf and insult_hits:
        markers.append("SLF_INSULT")

    for i, tok in enumerate(tokens):
        if tok in TARGET_TERMS or tok in SELF_TERMS:
            window = tokens[max(0, i - 2):min(len(tokens), i + 3)]
            prefix = "TGT_" if tok in TARGET_TERMS else "SLF_"
            for w in window:
                if w in NEGATIVE_WORDS:
                    markers.append(f"{prefix}NEG_{w}")
                if w in INSULT_TERMS:
                    markers.append(f"{prefix}{w}")

    if any(tok in NEGATIVE_WORDS for tok in tokens):
        markers.append("HAS_NEG")
    if any(tok in POSITIVE_WORDS for tok in tokens):
        markers.append("HAS_POS")
    if any(tok in INSULT_TERMS for tok in tokens):
        markers.append("HAS_INSULT")
    if "not" in tokens and any(tok in INSULT_TERMS for tok in tokens):
        markers.append("NEGATED_INSULT")
    if "!" in str(text):
        markers.append("EXCLAMATION")

    if not markers:
        markers.append("NO_SIGNAL")

    return " ".join(markers)

def min_distance(pos_a, pos_b, default=10):
    if not pos_a or not pos_b:
        return default
    return min(abs(a - b) for a in pos_a for b in pos_b)

# ============================================================
# 4) CLASS-AWARE LEXICON FEATURES  (from v2)
#    Demographic proximity: how close are age/gender/ethnicity/
#    religion terms to insult/negative words in the tweet.
# ============================================================
class ClassAwareLexiconFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=2):          # fixed: was _init_
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            arr = np.asarray(X)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            s = pd.Series(arr)
        return s.astype(str).fillna("")

    def transform(self, X):
        texts = self._to_series(X)
        rows = []

        for raw in texts:
            tokens = tokenize(raw)
            n = len(tokens)

            target_positions  = [i for i, t in enumerate(tokens) if t in TARGET_TERMS]
            self_positions    = [i for i, t in enumerate(tokens) if t in SELF_TERMS]
            insult_positions  = [i for i, t in enumerate(tokens) if t in INSULT_TERMS]
            neg_positions     = [i for i, t in enumerate(tokens) if t in NEGATIVE_WORDS]
            pos_positions     = [i for i, t in enumerate(tokens) if t in POSITIVE_WORDS]

            feat = []

            # global cue counts + minimum distances
            feat.append(len(target_positions))
            feat.append(len(self_positions))
            feat.append(len(insult_positions))
            feat.append(len(neg_positions))
            feat.append(len(pos_positions))
            feat.append(min_distance(target_positions, insult_positions))
            feat.append(min_distance(target_positions, neg_positions))
            feat.append(min_distance(self_positions, insult_positions))
            feat.append(min_distance(self_positions, neg_positions))

            # per-demographic-category proximity features
            for _, lex in DEMO_LEXICONS.items():
                lex_positions = [i for i, t in enumerate(tokens) if t in lex]
                count   = len(lex_positions)
                density = count / n if n else 0.0
                near_neg    = 0
                near_insult = 0
                near_pos    = 0

                for p in lex_positions:
                    lo = max(0, p - self.window_size)
                    hi = min(n, p + self.window_size + 1)
                    window = tokens[lo:hi]
                    near_neg    += sum(tok in NEGATIVE_WORDS for tok in window)
                    near_insult += sum(tok in INSULT_TERMS    for tok in window)
                    near_pos    += sum(tok in POSITIVE_WORDS  for tok in window)

                min_neg_dist    = min_distance(lex_positions, neg_positions,    default=10)
                min_insult_dist = min_distance(lex_positions, insult_positions, default=10)
                neutral_flag    = 1.0 if (count > 0 and near_neg == 0 and near_insult == 0) else 0.0

                feat.extend([
                    count,
                    density,
                    near_neg,
                    near_insult,
                    near_pos,
                    min_neg_dist,
                    min_insult_dist,
                    neutral_flag,
                ])

            # binary demographic presence flags
            feat.append(1.0 if any(t in GENDER_TERMS    for t in tokens) else 0.0)
            feat.append(1.0 if any(t in AGE_TERMS        for t in tokens) else 0.0)
            feat.append(1.0 if any(t in ETHNICITY_TERMS  for t in tokens) else 0.0)
            feat.append(1.0 if any(t in RELIGION_TERMS   for t in tokens) else 0.0)

            rows.append(feat)

        return np.asarray(rows, dtype=np.float32)


# ============================================================
# 5) CLASS CENTROID SIMILARITY  (from v3)
#    Soft prototype signal: cosine similarity to each class
#    TF-IDF centroid learned from training data.
# ============================================================
class ClassCentroidSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=30000):     # fixed: was _init_
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=max_features,
            sublinear_tf=True,
            stop_words="english",
        )

    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            arr = np.asarray(X)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            s = pd.Series(arr)
        return s.astype(str).fillna("")

    def fit(self, X, y):
        texts = self._to_series(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        X_vec = self.vectorizer.fit_transform(texts)

        self.centroids_ = {}
        for c in self.classes_:
            idx      = np.where(y == c)[0]
            centroid = np.asarray(X_vec[idx].mean(axis=0)).ravel()
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            self.centroids_[c] = centroid

        return self

    def transform(self, X):
        texts = self._to_series(X)
        X_vec = normalize(self.vectorizer.transform(texts), norm="l2", copy=False)

        feats = []
        for c in self.classes_:
            centroid = self.centroids_[c]
            dot      = X_vec @ centroid
            scores   = dot.A1 if hasattr(dot, "A1") else np.asarray(dot).ravel()
            feats.append(scores)

        return np.vstack(feats).T.astype(np.float32)


# ============================================================
# 6) NUMERIC STYLE / DENSITY FEATURES
# ============================================================
def build_numeric_features(text):
    raw    = str(text)
    norm   = normalize_text(raw)
    tokens = norm.split()

    char_count  = len(raw)
    word_count  = len(tokens)
    uniq_count  = len(set(tokens))
    alpha_count = sum(1 for c in raw if c.isalpha())
    upper_count = sum(1 for c in raw if c.isupper())
    digit_count = sum(1 for c in raw if c.isdigit())
    punct_count = sum(1 for c in raw if re.match(r"[^\w\s]", c))

    target_count = sum(t in TARGET_TERMS   for t in tokens)
    self_count   = sum(t in SELF_TERMS     for t in tokens)
    insult_count = sum(t in INSULT_TERMS   for t in tokens)
    neg_count    = sum(t in NEGATIVE_WORDS for t in tokens)
    pos_count    = sum(t in POSITIVE_WORDS for t in tokens)

    target_positions = [i for i, t in enumerate(tokens) if t in TARGET_TERMS]
    self_positions   = [i for i, t in enumerate(tokens) if t in SELF_TERMS]
    insult_positions = [i for i, t in enumerate(tokens) if t in INSULT_TERMS]
    neg_positions    = [i for i, t in enumerate(tokens) if t in NEGATIVE_WORDS]
    pos_positions    = [i for i, t in enumerate(tokens) if t in POSITIVE_WORDS]

    target_neg_window    = 0
    target_insult_window = 0
    for p in target_positions:
        lo     = max(0, p - 2)
        hi     = min(len(tokens), p + 3)
        window = tokens[lo:hi]
        target_neg_window    += sum(tok in NEGATIVE_WORDS for tok in window)
        target_insult_window += sum(tok in INSULT_TERMS   for tok in window)

    return pd.Series({
        "char_count":              char_count,
        "word_count":              word_count,
        "uniq_ratio":              uniq_count  / word_count  if word_count  else 0.0,
        "upper_ratio":             upper_count / alpha_count if alpha_count else 0.0,
        "digit_ratio":             digit_count / char_count  if char_count  else 0.0,
        "punct_ratio":             punct_count / char_count  if char_count  else 0.0,
        "exclam_count":            raw.count("!"),
        "question_count":          raw.count("?"),
        "elongated_count":         len(ELONG_RE.findall(raw)),
        "target_count":            target_count,
        "self_count":              self_count,
        "insult_count":            insult_count,
        "neg_count":               neg_count,
        "pos_count":               pos_count,
        "insult_density":          insult_count / word_count if word_count else 0.0,
        "neg_density":             neg_count    / word_count if word_count else 0.0,
        "pos_density":             pos_count    / word_count if word_count else 0.0,
        "tgt_insult_flag":         1 if target_count > 0 and insult_count > 0 else 0,
        "slf_insult_flag":         1 if self_count   > 0 and insult_count > 0 else 0,
        "target_insult_distance":  min_distance(target_positions, insult_positions),
        "target_neg_distance":     min_distance(target_positions, neg_positions),
        "target_pos_distance":     min_distance(target_positions, pos_positions),
        "self_insult_distance":    min_distance(self_positions,   insult_positions),
        "self_neg_distance":       min_distance(self_positions,   neg_positions),
        "target_neg_window":       target_neg_window,
        "target_insult_window":    target_insult_window,
        "obfuscation_count":       sum(ch in "@013457!$|" for ch in raw),
        "avg_token_len":           float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        "vader_compound":          analyzer.polarity_scores(raw)["compound"],
        "vader_neg":               analyzer.polarity_scores(raw)["neg"],
        "vader_pos":               analyzer.polarity_scores(raw)["pos"],
    })


# ============================================================
# 7) BUILD DATASET
# ============================================================
print("Building features...")
df_clean["clean_text"]       = df_clean["tweet_text"].apply(normalize_text)
df_clean["masked_text"]      = df_clean["tweet_text"].apply(mask_structural)
df_clean["interaction_text"] = df_clean["tweet_text"].apply(interaction_tokens)

num_df   = df_clean["tweet_text"].apply(build_numeric_features)
df_clean = pd.concat([df_clean.reset_index(drop=True), num_df.reset_index(drop=True)], axis=1)

numeric_cols = list(num_df.columns)

X = df_clean[["tweet_text", "clean_text", "masked_text", "interaction_text"] + numeric_cols].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_clean["cyberbullying_type"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

n_classes = len(label_encoder.classes_)

# ============================================================
# 8) FEATURE EXTRACTORS
# ============================================================
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    max_features=40000,
    sublinear_tf=True,
    stop_words="english",
)

char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=2,
    max_features=40000,
    sublinear_tf=True,
)

mask_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    max_features=20000,
    sublinear_tf=True,
)

interaction_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    max_features=12000,
    sublinear_tf=True,
)

class_aware_lex  = ClassAwareLexiconFeatures(window_size=2)
class_centroid   = ClassCentroidSimilarity(max_features=30000)

preprocessor = ColumnTransformer(
    transformers=[
        ("word",      word_vec,        "clean_text"),
        ("char",      char_vec,        "clean_text"),
        ("mask",      mask_vec,        "masked_text"),
        ("inter",     interaction_vec, "interaction_text"),
        ("classlex",  class_aware_lex, "tweet_text"),   # v2: demographic proximity
        ("classcent", class_centroid,  "tweet_text"),   # v3: prototype similarity
        ("num",       MinMaxScaler(),  numeric_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

# ============================================================
# 9) MODEL FACTORY
# ============================================================
def make_meta_model(use_gpu=True):
    common = dict(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=4,
    )
    if use_gpu:
        try:
            return XGBClassifier(**common, tree_method="gpu_hist", predictor="gpu_predictor")
        except Exception:
            try:
                return XGBClassifier(**common, tree_method="hist", device="cuda")
            except Exception:
                pass
    try:
        return XGBClassifier(**common, tree_method="hist")
    except Exception:
        return LogisticRegression(
            C=0.5, class_weight="balanced",
            max_iter=5000, solver="lbfgs", random_state=42,
        )

def build_pipeline(use_gpu=True):
    base_estimators = [
        (
            "svc",
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=0.75, class_weight="balanced",
                    dual=False, random_state=42,
                ),
                cv=3, method="sigmoid",
            ),
        ),
        (
            "logreg",
            LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=5000, solver="lbfgs", random_state=42,
            ),
        ),
        (
            "nb",
            ComplementNB(alpha=0.15),
        ),
    ]

    stacked_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=make_meta_model(use_gpu=use_gpu),
        cv=3,
        stack_method="auto",
        n_jobs=-1,
        passthrough=False,
    )

    return Pipeline([
        ("features", preprocessor),
        ("clf",      stacked_model),
    ])

# ============================================================
# 10) TRAIN / CACHE
# ============================================================
import __main__
__main__.ClassAwareLexiconFeatures = ClassAwareLexiconFeatures
__main__.ClassCentroidSimilarity = ClassCentroidSimilarity

MODEL_CACHE_FILE = "ml2_pipeline.joblib"
LABEL_ENCODER_FILE = "ml2_label_encoder.joblib"

pipeline = None
if os.path.exists(MODEL_CACHE_FILE) and os.path.exists(LABEL_ENCODER_FILE):
    print("Loading cached model from disk... (Instant startup!)")
    pipeline = joblib.load(MODEL_CACHE_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
else:
    try:
        print("Training on GPU path (Initial Run)...")
        pipeline = build_pipeline(use_gpu=True)
        pipeline.fit(X_train, y_train)
        print("GPU training complete.")
    except Exception as e:
        print(f"GPU path failed ({e}), falling back to CPU...")
        pipeline = build_pipeline(use_gpu=False)
        pipeline.fit(X_train, y_train)
        print("CPU training complete.")
        
    print("Saving trained model to disk for future speedy loads...")
    joblib.dump(pipeline, MODEL_CACHE_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)

# ============================================================
# 11) EVALUATE
# ============================================================
y_pred = pipeline.predict(X_test)

y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

acc = accuracy_score(y_test_labels, y_pred_labels)

print("\n" + "=" * 70)
print(f"ACCURACY: {acc * 100:.2f}%")
print("=" * 70 + "\n")
print(classification_report(y_test_labels, y_pred_labels, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test_labels, y_pred_labels))
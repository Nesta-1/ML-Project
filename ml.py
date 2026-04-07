import re
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
df=pd.read_csv("cyberbullying_tweets.csv")
# ============================================================
# 1) DATA CLEANING
# ============================================================
required_cols = {"tweet_text", "cyberbullying_type"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")


df_clean = df.drop_duplicates(subset=["tweet_text"]).copy()
df_clean = df_clean[df_clean["cyberbullying_type"] != "other_cyberbullying"].copy()
df_clean = df_clean.reset_index(drop=True)

# ============================================================
# 2) NOVEL FEATURE LEXICONS
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
    "$": "s"
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

def mask_structural(text):
    text = normalize_text(text)
    tokens = text.split()
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
    text = normalize_text(text)
    tokens = text.split()

    markers = []
    tgt_present = any(t in TARGET_TERMS for t in tokens)
    slf_present = any(t in SELF_TERMS for t in tokens)
    insult_hits = [t for t in tokens if t in INSULT_TERMS]

    if tgt_present and insult_hits:
        markers.append("TGT_INSULT")
    if slf_present and insult_hits:
        markers.append("SLF_INSULT")

    # local target-insult / self-insult windows
    for i, tok in enumerate(tokens):
        if tok in TARGET_TERMS or tok in SELF_TERMS:
            window = tokens[max(0, i - 2):min(len(tokens), i + 3)]
            prefix = "TGT_" if tok in TARGET_TERMS else "SLF_"
            for w in window:
                if w in INSULT_TERMS:
                    markers.append(f"{prefix}{w}")

    # simple abuse patterns
    if any(tok in INSULT_TERMS for tok in tokens):
        markers.append("HAS_INSULT")
    if "not" in tokens and any(tok in INSULT_TERMS for tok in tokens):
        markers.append("NEGATED_INSULT")
    if "!" in str(text):
        markers.append("EXCLAMATION")

    if not markers:
        markers.append("NO_INTERACTION")
    return " ".join(markers)

# ============================================================
# 4) NUMERIC STYLE / DENSITY FEATURES
# ============================================================
def build_numeric_features(text):
    raw = str(text)
    norm = normalize_text(raw)
    tokens = norm.split()

    char_count = len(raw)
    word_count = len(tokens)
    uniq_count = len(set(tokens))

    alpha_count = sum(1 for c in raw if c.isalpha())
    upper_count = sum(1 for c in raw if c.isupper())
    digit_count = sum(1 for c in raw if c.isdigit())
    punct_count = sum(1 for c in raw if re.match(r"[^\w\s]", c))

    target_count = sum(t in TARGET_TERMS for t in tokens)
    self_count = sum(t in SELF_TERMS for t in tokens)
    insult_count = sum(t in INSULT_TERMS for t in tokens)

    return pd.Series({
        "char_count": char_count,
        "word_count": word_count,
        "uniq_ratio": uniq_count / word_count if word_count else 0.0,
        "upper_ratio": upper_count / alpha_count if alpha_count else 0.0,
        "digit_ratio": digit_count / char_count if char_count else 0.0,
        "punct_ratio": punct_count / char_count if char_count else 0.0,
        "exclam_count": raw.count("!"),
        "question_count": raw.count("?"),
        "elongated_count": len(ELONG_RE.findall(raw)),
        "target_count": target_count,
        "self_count": self_count,
        "insult_count": insult_count,
        "tgt_insult_flag": 1 if target_count > 0 and insult_count > 0 else 0,
        "slf_insult_flag": 1 if self_count > 0 and insult_count > 0 else 0,
        "obfuscation_count": sum(ch in "@013457!$|" for ch in raw),
        "avg_token_len": float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
    })

# ============================================================
# 5) PREPARE DATAFRAME
# ============================================================
print("Building features...")
df_clean["clean_text"] = df_clean["tweet_text"].apply(normalize_text)
df_clean["masked_text"] = df_clean["tweet_text"].apply(mask_structural)
df_clean["interaction_text"] = df_clean["tweet_text"].apply(interaction_tokens)

num_df = df_clean["tweet_text"].apply(build_numeric_features)
df_clean = pd.concat([df_clean.reset_index(drop=True), num_df.reset_index(drop=True)], axis=1)

numeric_cols = list(num_df.columns)

X = df_clean[["clean_text", "masked_text", "interaction_text"] + numeric_cols].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_clean["cyberbullying_type"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

n_classes = len(label_encoder.classes_)

# ============================================================
# 6) FEATURE EXTRACTOR
# ============================================================
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    max_features=50000,
    sublinear_tf=True,
    stop_words="english"
)

char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

mask_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    max_features=25000,
    sublinear_tf=True
)

interaction_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1,
    max_features=8000,
    sublinear_tf=True
)

preprocessor = ColumnTransformer(
    transformers=[
        ("word", word_vec, "clean_text"),
        ("char", char_vec, "clean_text"),
        ("mask", mask_vec, "masked_text"),
        ("inter", interaction_vec, "interaction_text"),
        ("num", MinMaxScaler(), numeric_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3
)

# ============================================================
# 7) MODEL FACTORY
# ============================================================
def make_xgb(use_gpu=True):
    common = dict(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=4
    )

    if use_gpu:
        # XGBoost >= 2.0: use device="cuda" with tree_method="hist"
        return XGBClassifier(**common, tree_method="hist", device="cuda")

    return XGBClassifier(**common, tree_method="hist")

def build_pipeline(use_gpu=True):
    base_estimators = [
        (
            "svc",
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=0.75,
                    class_weight="balanced",
                    dual=False,
                    random_state=42
                ),
                cv=3,
                method="sigmoid"
            )
        ),
        (
            "logreg",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=42
            )
        ),
        (
            "nb",
            ComplementNB(alpha=0.2)
        )
    ]

    meta_model = make_xgb(use_gpu=use_gpu)

    stacked_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=3,
        stack_method="auto",
        n_jobs=4,
        passthrough=False
    )

    return ImbPipeline([
        ("features", preprocessor),
        ("sampler", RandomOverSampler(random_state=42)),
        ("clf", stacked_model)
    ])

# ============================================================
# 8) TRAIN
# ============================================================
pipeline = None
try:
    print("Training on GPU path (XGBoost device=cuda)...")
    pipeline = build_pipeline(use_gpu=True)
    print("Pipeline built. Fitting... (this may take a few minutes with stacking)")
    pipeline.fit(X_train, y_train)
    print("GPU training complete!")
except Exception as e:
    print(f"GPU path failed, falling back to CPU. Reason: {e}")
    pipeline = build_pipeline(use_gpu=False)
    print("CPU pipeline built. Fitting...")
    pipeline.fit(X_train, y_train)
    print("CPU training complete!")

# ============================================================
# 9) EVALUATE
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
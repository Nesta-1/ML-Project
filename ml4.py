import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, normalize, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# ============================================================
# 1) DATA CHECKS + CLEANING (6-Class, Duplicates Kept)
# ============================================================
# Assuming 'df' is already loaded in your environment. 
# If not, add your pd.read_csv() line here.
df_clean = df.copy().reset_index(drop=True)

print(f"Dataset: {len(df_clean)} rows, {df_clean['cyberbullying_type'].nunique()} classes")
print(df_clean["cyberbullying_type"].value_counts().to_string())
print()

# ============================================================
# 2) LEXICONS & HELPERS
# ============================================================
TARGET_TERMS = {"you", "u", "ur", "your", "yours", "yourself", "yall", "ya", "they", "them", "their", "he", "she", "him", "her"}
SELF_TERMS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
INSULT_TERMS = {"idiot", "stupid", "dumb", "loser", "bitch", "bastard", "trash", "ugly", "hate", "pathetic", "fool", "shut", "annoying", "moron", "asshole", "jerk", "freak", "worthless", "disgusting", "clown", "brat", "suck"}
NEGATIVE_WORDS = INSULT_TERMS 
POSITIVE_WORDS = {"good", "great", "nice", "beautiful", "smart", "cool", "amazing", "kind", "love", "respect", "sweet", "pretty"}
DEMO_LEXICONS = {
    "age": {"age", "aged", "old", "older", "young", "younger", "kid", "kids", "child", "teen", "teenager", "adult", "boy", "girl", "baby", "grandpa", "grandma"},
    "gender": {"gender", "boy", "girl", "man", "men", "woman", "women", "male", "female", "guy", "guys", "trans", "transgender"},
    "ethnicity": {"black", "white", "asian", "african", "european", "indian", "arab", "latino", "hispanic", "chinese", "japanese", "race", "ethnic"},
    "religion": {"religion", "religious", "muslim", "islam", "christian", "hindu", "jew", "jewish", "buddhist", "sikh", "atheist", "church", "mosque", "temple", "bible", "quran", "god", "allah"}
}
OBFUSCATION_MAP = str.maketrans({"@": "a", "0": "o", "1": "i", "!": "i", "|": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})
URL_RE, MENTION_RE, PUNCT_RE, WS_RE, ELONG_RE = re.compile(r"http\S+|www\S+"), re.compile(r"@\w+"), re.compile(r"[^\w\s]"), re.compile(r"\s+"), re.compile(r"(.)\1{2,}")

def normalize_text(text):
    text = str(text).lower()
    text = URL_RE.sub(" ", text).replace("#", " ").translate(OBFUSCATION_MAP)
    return WS_RE.sub(" ", PUNCT_RE.sub(" ", text)).strip()

def tokenize(text): return normalize_text(text).split()

def mask_structural(text):
    return " ".join(["TGT" if t in TARGET_TERMS else "SLF" if t in SELF_TERMS else t for t in tokenize(text)])

def interaction_tokens(text):
    tokens = tokenize(text)
    markers = []
    if any(t in TARGET_TERMS for t in tokens) and any(t in INSULT_TERMS for t in tokens): markers.append("TGT_INSULT")
    if "!" in str(text): markers.append("EXCLAMATION")
    return " ".join(markers) if markers else "NO_SIGNAL"

def min_dist(pos_a, pos_b): return min(abs(a - b) for a in pos_a for b in pos_b) if pos_a and pos_b else 10

# ============================================================
# 3) CUSTOM TRANSFORMERS (With memory-safe flatten fix)
# ============================================================
class ClassAwareLexiconFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows = []
        for raw in X:
            tokens = tokenize(raw); n = len(tokens)
            t_p = [i for i, t in enumerate(tokens) if t in TARGET_TERMS]
            i_p = [i for i, t in enumerate(tokens) if t in INSULT_TERMS]
            feat = [len(t_p), len(i_p), min_dist(t_p, i_p)]
            for _, lex in DEMO_LEXICONS.items():
                l_p = [i for i, t in enumerate(tokens) if t in lex]
                feat.extend([len(l_p), len(l_p)/n if n else 0, min_dist(l_p, i_p)])
            rows.append(feat)
        return np.asarray(rows, dtype=np.float32)

class ClassCentroidSimilarity(BaseEstimator, TransformerMixin):
    def _init_(self, max_features=10000):
        self.max_features = max_features
        self.vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words="english")
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_vec = self.vec.fit_transform(X)
        self.centroids_ = {}
        for c in self.classes_:
            centroid = np.asarray(X_vec[y == c].mean(axis=0)).flatten()
            self.centroids_[c] = centroid / (np.linalg.norm(centroid) + 1e-12)
        return self
        
    def transform(self, X):
        X_vec = normalize(self.vec.transform(X))
        feats = []
        for c in self.classes_:
            dot = X_vec @ self.centroids_[c]
            scores = np.asarray(dot).flatten()
            feats.append(scores)
        return np.vstack(feats).T.astype(np.float32)

def build_numeric_features(text):
    raw = str(text)
    return pd.Series({"char_count": len(raw), "upper_ratio": sum(1 for c in raw if c.isupper())/len(raw) if len(raw) else 0,
                      "obfuscation_count": sum(ch in "@013457!$|" for ch in raw), "avg_token_len": np.mean([len(t) for t in tokenize(raw)]) if tokenize(raw) else 0})

# ============================================================
# 4) PIPELINE PREPARATION (Clamped for RTX 3050 6GB VRAM)
# ============================================================
print("\nEngineering features...")
df_clean["clean_text"] = df_clean["tweet_text"].apply(normalize_text)
df_clean["masked_text"] = df_clean["tweet_text"].apply(mask_structural)
df_clean["inter_text"] = df_clean["tweet_text"].apply(interaction_tokens)
num_df = df_clean["tweet_text"].apply(build_numeric_features)
df_clean = pd.concat([df_clean, num_df], axis=1)

X = df_clean[["tweet_text", "clean_text", "masked_text", "inter_text"] + list(num_df.columns)]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_clean["cyberbullying_type"])
n_classes = len(label_encoder.classes_)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full)

# Feature limits clamped to prevent 6GB Out Of Memory Error
preprocessor = ColumnTransformer([
    ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=10000, sublinear_tf=True, stop_words="english"), "clean_text"),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=10000), "clean_text"),
    ("mask", TfidfVectorizer(max_features=5000), "masked_text"),
    ("cent", ClassCentroidSimilarity(max_features=10000), "tweet_text"),
    ("lex", ClassAwareLexiconFeatures(), "tweet_text"),
    ("num", MinMaxScaler(), list(num_df.columns))
])

print("Fitting Pipeline... (Building the feature matrix)")
X_train_f = preprocessor.fit_transform(X_train, y_train)
X_val_f = preprocessor.transform(X_val)
X_test_f = preprocessor.transform(X_test)

# ============================================================
# 5) XGBOOST (FAST EARLY STOPPING ENABLED)
# ============================================================
print("\nTraining main model with FAST early stopping...")
sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

# Early stopping is properly placed in the constructor. 
# It will stop as soon as validation loss stops improving.
clf = XGBClassifier(
    objective="multi:softprob", 
    num_class=n_classes, 
    tree_method="hist", 
    device="cuda", 
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    early_stopping_rounds=40,
    random_state=42
)

clf.fit(
    X_train_f, y_train, 
    sample_weight=sample_weight, 
    eval_set=[(X_train_f, y_train), (X_val_f, y_val)], 
    verbose=False
)
print(f"Main model training finished early at tree {clf.best_iteration}!")

# ============================================================
# 6) EVALUATION & ARTIFACT SAVING
# ============================================================
y_pred = clf.predict(X_test_f)
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\n" + "=" * 70)
print(f"FINAL 6-CLASS TEST SET ACCURACY: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("=" * 70)
print(classification_report(y_test_labels, y_pred_labels, digits=4))

joblib.dump(preprocessor, 'preprocessor.pkl')
clf.save_model('cyberbullying_xgb.json')

# ============================================================
# 7) VISUALIZATIONS
# ============================================================
print("\nGenerating 19 Visualizations...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Raw Confusion Matrix'); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig("01_CM_Raw.png", dpi=300); plt.close()

plt.figure(figsize=(8, 6)); sns.heatmap(cm/cm.sum(axis=1)[:,None], annot=True, fmt='.2f', cmap='magma', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Normalized Confusion Matrix (%)'); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig("02_CM_Normalized.png", dpi=300); plt.close()

try:
    res = clf.evals_result()
    plt.figure(figsize=(10, 6)); plt.plot(res['validation_0']['mlogloss'], label='Train Loss'); plt.plot(res['validation_1']['mlogloss'], label='Validation Loss')
    plt.title('XGBoost Multi-Class LogLoss Learning Curve'); plt.xlabel('Trees'); plt.legend(); plt.tight_layout(); plt.savefig("03_Learning_Curve.png", dpi=300); plt.close()
except: pass

imp_gain = sorted(clf.get_booster().get_score(importance_type="gain").items(), key=lambda x: x[1], reverse=True)[:20]
plt.figure(figsize=(10, 8)); sns.barplot(x=[x[1] for x in imp_gain], y=[x[0] for x in imp_gain], palette="viridis")
plt.title("Top 20 Features (Gain)"); plt.tight_layout(); plt.savefig("04_Feature_Importance_Gain.png", dpi=300); plt.close()

imp_wt = sorted(clf.get_booster().get_score(importance_type="weight").items(), key=lambda x: x[1], reverse=True)[:20]
plt.figure(figsize=(10, 8)); sns.barplot(x=[x[1] for x in imp_wt], y=[x[0] for x in imp_wt], palette="rocket")
plt.title("Top 20 Features (Weight)"); plt.tight_layout(); plt.savefig("05_Feature_Importance_Weight.png", dpi=300); plt.close()

y_test_bin = label_binarize(y_test, classes=range(n_classes))
y_proba = clf.predict_proba(X_test_f)

plt.figure(figsize=(9, 7))
for i, color in zip(range(n_classes), sns.color_palette("husl", n_classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{label_encoder.classes_[i]} (AUC={auc(fpr, tpr):.3f})')
plt.plot([0, 1], [0, 1], 'k--'); plt.legend(loc="lower right"); plt.title('ROC Curves'); plt.tight_layout(); plt.savefig("06_ROC_Curves.png", dpi=300); plt.close()

plt.figure(figsize=(9, 7))
for i, color in zip(range(n_classes), sns.color_palette("husl", n_classes)):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(rec, prec, color=color, lw=2.5, label=f'{label_encoder.classes_[i]} (AP={average_precision_score(y_test_bin[:, i], y_proba[:, i]):.3f})')
plt.legend(loc="lower left"); plt.title('Precision-Recall Curves'); plt.tight_layout(); plt.savefig("07_PR_Curves.png", dpi=300); plt.close()

svd = TruncatedSVD(n_components=2, random_state=42)
X_test_2d = svd.fit_transform(X_test_f)
plt.figure(figsize=(10, 8)); sns.scatterplot(x=X_test_2d[:, 0], y=X_test_2d[:, 1], hue=y_test_labels, palette="tab10", s=15, alpha=0.7)
plt.title("PCA Feature Space Projection"); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(); plt.savefig("08_PCA_Scatter.png", dpi=300); plt.close()

plt.figure(figsize=(8, 5)); sns.countplot(y=df_clean["cyberbullying_type"], palette="viridis")
plt.title("Class Distribution"); plt.tight_layout(); plt.savefig("09_Class_Dist.png", dpi=300); plt.close()

plt.figure(figsize=(10, 6)); sns.kdeplot(data=df_clean, x="char_count", hue="cyberbullying_type", fill=True, palette="tab10")
plt.xlim(0, df_clean["char_count"].quantile(0.99)); plt.title("Tweet Character Count"); plt.tight_layout(); plt.savefig("10_Length_KDE.png", dpi=300); plt.close()

plt.figure(figsize=(10, 8)); sns.heatmap(num_df.corr(), cmap="coolwarm", center=0); plt.title("Numeric Feature Correlation"); plt.tight_layout(); plt.savefig("11_Correlation.png", dpi=300); plt.close()

correct = (y_test == y_pred); conf = np.max(y_proba, axis=1)
plt.figure(figsize=(9, 6)); sns.histplot(conf[correct], color="green", label="Correct", kde=True, alpha=0.5); sns.histplot(conf[~correct], color="red", label="Incorrect", kde=True, alpha=0.5)
plt.title("Prediction Confidence"); plt.legend(); plt.tight_layout(); plt.savefig("12_Confidence.png", dpi=300); plt.close()

try:
    import scipy.sparse as sp
    X_sample = X_test_f[:300].toarray() if sp.issparse(X_test_f) else X_test_f[:300]
    names = [n.split('__')[-1] for n in preprocessor.get_feature_names_out()]
    shap_vals = shap.TreeExplainer(clf).shap_values(X_sample)
    plt.figure(figsize=(12, 8)); shap.summary_plot(shap_vals, X_sample, feature_names=names, show=False, class_names=label_encoder.classes_)
    plt.title("SHAP Feature Impact"); plt.tight_layout(); plt.savefig("13_SHAP.png", dpi=300, bbox_inches='tight'); plt.close()
except: pass

for i, label in enumerate(label_encoder.classes_):
    text = " ".join(df_clean[df_clean["cyberbullying_type"] == label]["clean_text"])
    plt.figure(figsize=(8, 4)); plt.imshow(WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text))
    plt.axis('off'); plt.title(f"Frequent Words: {label.upper()}"); plt.tight_layout(); plt.savefig(f"{14+i}WordCloud{label}.png", dpi=300); plt.close()

print("\nSUCCESS! Check your local folder for the images and models. You are done.")
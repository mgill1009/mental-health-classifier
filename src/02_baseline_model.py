"""
02_baseline_model.py
Trains a TF-IDF + Logistic Regression pipeline on the processed training data.
Outputs: models/baseline/pipeline.pkl, baseline_metrics.json, tfidf_feature_names.npy
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("models/baseline")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
train = pd.read_csv(PROCESSED_DIR / "train.csv")
val   = pd.read_csv(PROCESSED_DIR / "val.csv")
test  = pd.read_csv(PROCESSED_DIR / "test.csv")

X_train, y_train = train["text_tfidf"].fillna(""), train["label"]
X_val,   y_val   = val["text_tfidf"].fillna(""),   val["label"]
X_test,  y_test  = test["text_tfidf"].fillna(""),  test["label"]

weights = np.load(PROCESSED_DIR / "class_weights.npy")
class_weight_dict = {i: w for i, w in enumerate(weights)}

# ── Pipeline ──────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidfvectorizer", TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )),
    ("logisticregression", LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weight_dict,
        random_state=42,
    )),
])

# ── Train ─────────────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────
def evaluate(X, y, split):
    preds = pipeline.predict(X)
    return {
        "split":    split,
        "accuracy": round(float((preds == y).mean()), 4),
        "macro_f1": round(float(f1_score(y, preds, average="macro")), 4),
        "report":   classification_report(y, preds, target_names=["Low","Moderate","High"], output_dict=True),
    }

metrics = {
    "train": evaluate(X_train, y_train, "train"),
    "val":   evaluate(X_val,   y_val,   "val"),
    "test":  evaluate(X_test,  y_test,  "test"),
}

# ── Save ──────────────────────────────────────────────────────────────────
joblib.dump(pipeline, MODEL_DIR / "pipeline.pkl")

feature_names = pipeline.named_steps["tfidfvectorizer"].get_feature_names_out()
np.save(MODEL_DIR / "tfidf_feature_names.npy", feature_names)

with open(MODEL_DIR / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(
    f"Done. "
    f"Train macro-F1={metrics['train']['macro_f1']} | "
    f"Val macro-F1={metrics['val']['macro_f1']} | "
    f"Test macro-F1={metrics['test']['macro_f1']}"
)

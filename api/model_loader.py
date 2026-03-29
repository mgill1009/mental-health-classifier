"""
model_loader.py — singleton model cache for FastAPI serving layer.

Exports:
  load_tfidf()       — load TF-IDF pipeline into memory (idempotent)
  load_distilbert()  — load DistilBERT model into memory (idempotent)
  predict_tfidf(text)      -> dict
  predict_distilbert(text) -> dict
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
# Files land at /app/models/ inside the container (WORKDIR is /app)
TFIDF_PATH      = os.environ.get("TFIDF_PATH",      "models/baseline/pipeline.pkl")
DISTILBERT_PATH = os.environ.get("DISTILBERT_PATH", "models/distilbert/distilbert-mental-health")
MAX_LENGTH = 256

RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

# ── Singletons ────────────────────────────────────────────────────────────
_tfidf_pipeline       = None
_distilbert_model     = None
_distilbert_tokenizer = None
_device               = None


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {_device}")
    return _device


# ── Loaders ───────────────────────────────────────────────────────────────
def load_tfidf():
    global _tfidf_pipeline
    if _tfidf_pipeline is None:
        import joblib
        logger.info(f"Loading TF-IDF pipeline from {TFIDF_PATH}")
        _tfidf_pipeline = joblib.load(TFIDF_PATH)
        logger.info("TF-IDF pipeline loaded.")
    return _tfidf_pipeline


def load_distilbert():
    global _distilbert_model, _distilbert_tokenizer
    if _distilbert_model is None:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        logger.info(f"Loading DistilBERT from {DISTILBERT_PATH}")
        _distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        _distilbert_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
        _distilbert_model.to(_get_device())
        _distilbert_model.eval()
        logger.info("DistilBERT loaded.")
    return _distilbert_tokenizer, _distilbert_model


# ── Inference ─────────────────────────────────────────────────────────────
def predict_tfidf(text: str) -> dict:
    pipeline = load_tfidf()
    probs = pipeline.predict_proba([text])[0]
    label = int(np.argmax(probs))
    return {
        "risk_tier":     label,
        "risk_label":    RISK_LABELS[label],
        "confidence":    round(float(probs[label]), 4),
        "probabilities": {RISK_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def predict_distilbert(text: str) -> dict:
    import torch
    tokenizer, model = load_distilbert()
    device = _get_device()
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    label = int(np.argmax(probs))
    return {
        "risk_tier":     label,
        "risk_label":    RISK_LABELS[label],
        "confidence":    round(float(probs[label]), 4),
        "probabilities": {RISK_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }

"""
model_loader.py
Loads and caches both models at startup.
Keeps models in memory so they are not reloaded on every request.
"""

import os
import time
import joblib
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
BASELINE_DIR = BASE_DIR / "models" / "baseline"
DISTILBERT_DIR = BASE_DIR / "models" / "distilbert" / "distilbert-mental-health"

# Allow overriding via env vars (useful for Cloud Run / Cloud Functions)
BASELINE_PATH    = Path(os.getenv("BASELINE_PATH",    str(BASELINE_DIR / "pipeline.pkl")))
DISTILBERT_PATH  = Path(os.getenv("DISTILBERT_PATH",  str(DISTILBERT_DIR)))

# ── Label map ─────────────────────────────────────────────────────────────
RISK_NAMES = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

# ── Model registry ────────────────────────────────────────────────────────
_models: dict = {}


def load_baseline() -> dict:
    """Load TF-IDF + LogReg sklearn pipeline from disk."""
    t0 = time.perf_counter()
    pipeline = joblib.load(BASELINE_PATH)
    load_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[baseline] Loaded in {load_ms:.1f}ms from {BASELINE_PATH}")
    return {"pipeline": pipeline, "load_ms": load_ms}


def load_distilbert() -> dict:
    """Load fine-tuned DistilBERT model and tokenizer from disk."""
    # Lazy import — only pull in torch/transformers if distilbert is needed
    import torch
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
    )

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(DISTILBERT_PATH))
    model     = DistilBertForSequenceClassification.from_pretrained(str(DISTILBERT_PATH))
    model.to(device)
    model.eval()

    load_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[distilbert] Loaded in {load_ms:.1f}ms — device={device}")
    return {"model": model, "tokenizer": tokenizer, "device": device, "load_ms": load_ms}


def get_model(name: str) -> dict:
    """
    Return cached model dict. Loads on first call (lazy singleton).
    name: 'tfidf' | 'distilbert'
    """
    if name not in _models:
        if name == "tfidf":
            _models["tfidf"] = load_baseline()
        elif name == "distilbert":
            _models["distilbert"] = load_distilbert()
        else:
            raise ValueError(f"Unknown model '{name}'. Choose 'tfidf' or 'distilbert'.")
    return _models[name]


def preload_all():
    """Eagerly load both models at startup (used by Cloud Run / uvicorn)."""
    logger.info("Preloading all models...")
    get_model("tfidf")
    get_model("distilbert")
    logger.info("All models ready.")


# ── Inference helpers ──────────────────────────────────────────────────────

def predict_tfidf(text: str) -> dict:
    """Run inference with TF-IDF + LogReg pipeline."""
    m = get_model("tfidf")
    pipeline = m["pipeline"]

    t0 = time.perf_counter()
    proba     = pipeline.predict_proba([text])[0]
    label     = int(np.argmax(proba))
    latency   = (time.perf_counter() - t0) * 1000

    return {
        "risk_tier":  label,
        "risk_label": RISK_NAMES[label],
        "confidence": round(float(proba[label]), 4),
        "probabilities": {RISK_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "model":      "tfidf",
        "latency_ms": round(latency, 3),
    }


def predict_distilbert(text: str, max_length: int = 256) -> dict:
    """Run inference with fine-tuned DistilBERT."""
    import torch
    import torch.nn.functional as F

    m         = get_model("distilbert")
    model     = m["model"]
    tokenizer = m["tokenizer"]
    device    = m["device"]

    t0 = time.perf_counter()
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        proba  = F.softmax(logits, dim=-1)[0].cpu().numpy()

    label   = int(np.argmax(proba))
    latency = (time.perf_counter() - t0) * 1000

    return {
        "risk_tier":  label,
        "risk_label": RISK_NAMES[label],
        "confidence": round(float(proba[label]), 4),
        "probabilities": {RISK_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "model":      "distilbert",
        "latency_ms": round(latency, 3),
    }

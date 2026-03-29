"""
main.py — FastAPI serving layer for Mental Health Risk Classifier.

Supports three deployment modes via MODEL_TYPE environment variable:
  MODEL_TYPE=both        load both models at startup (Cloud Run default)
  MODEL_TYPE=tfidf       load TF-IDF only (lightweight K8s pod)
  MODEL_TYPE=distilbert  load DistilBERT only (heavy K8s pod)

Endpoints:
  GET  /health           liveness + readiness probe
  GET  /labels           risk tier label map
  POST /predict          inference  (?model=tfidf|distilbert)
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_loader import predict_tfidf, predict_distilbert, load_tfidf, load_distilbert

# ── Config ────────────────────────────────────────────────────────────────
MODEL_TYPE = os.environ.get("MODEL_TYPE", "both").lower()
# Valid: "tfidf" | "distilbert" | "both"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

# ── Startup ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Preload only the model(s) this instance is configured to serve.
    Lazy loading would charge cold-start time to the first request —
    eager loading here moves that cost to pod startup, which is the
    correct trade-off for Cloud Run and GKE.
    Cloud Functions uses a separate entry point with lazy loading instead.
    """
    logger.info(f"Startup — MODEL_TYPE={MODEL_TYPE}")
    if MODEL_TYPE in ("tfidf", "both"):
        t0 = time.perf_counter()
        load_tfidf()
        logger.info(f"TF-IDF ready in {(time.perf_counter()-t0)*1000:.0f}ms")
    if MODEL_TYPE in ("distilbert", "both"):
        t0 = time.perf_counter()
        load_distilbert()
        logger.info(f"DistilBERT ready in {(time.perf_counter()-t0)*1000:.0f}ms")
    yield
    logger.info("Shutdown.")

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Risk Classifier",
    version="2.0.0",
    description="Containerized vs Serverless ML serving — CMPT 756",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    risk_tier: int
    risk_label: str
    confidence: float
    probabilities: dict
    model: str
    latency_ms: float

# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Kubernetes liveness + readiness probe target."""
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "serving": (
            ["tfidf", "distilbert"] if MODEL_TYPE == "both"
            else [MODEL_TYPE]
        ),
    }

@app.get("/labels")
def labels():
    return RISK_LABELS

@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    model: str = Query("tfidf", enum=["tfidf", "distilbert"]),
):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    # Reject requests for a model this pod doesn't serve
    if MODEL_TYPE != "both" and model != MODEL_TYPE:
        raise HTTPException(
            status_code=400,
            detail=f"This instance serves MODEL_TYPE={MODEL_TYPE}. "
                   f"Route your request to the '{model}' service.",
        )

    t0 = time.perf_counter()
    result = predict_tfidf(body.text) if model == "tfidf" else predict_distilbert(body.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictResponse(
        risk_tier=result["risk_tier"],
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model=model,
        latency_ms=latency_ms,
    )
@app.post("/tfidf/predict")
def predict_tfidf_alias(body: PredictRequest):
    """Alias for GKE Ingress routing — forwards /tfidf/predict to tfidf model."""
    t0 = time.perf_counter()
    result = predict_tfidf(body.text)
    return PredictResponse(
        risk_tier=result["risk_tier"],
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model="tfidf",
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

@app.post("/distilbert/predict")
def predict_distilbert_alias(body: PredictRequest):
    """Alias for GKE Ingress routing — forwards /distilbert/predict to distilbert model."""
    t0 = time.perf_counter()
    result = predict_distilbert(body.text)
    return PredictResponse(
        risk_tier=result["risk_tier"],
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model="distilbert",
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
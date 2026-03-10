"""
main.py
FastAPI application serving both TF-IDF and DistilBERT inference endpoints.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Endpoints:
    GET  /              health check
    GET  /health        detailed health + model status
    POST /predict       inference (query param: model=tfidf|distilbert)
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_loader import predict_tfidf, predict_distilbert, get_model, RISK_NAMES

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Startup / shutdown ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Preload both models on startup so the first request is not penalised.
    Comment out preload_all() to use lazy loading instead
    (better for Cloud Functions where startup time is billed).
    """
    logger.info("Starting up — preloading models...")
    try:
        get_model("tfidf")
        logger.info("TF-IDF model ready.")
    except Exception as e:
        logger.warning(f"TF-IDF preload failed: {e}")
    try:
        get_model("distilbert")
        logger.info("DistilBERT model ready.")
    except Exception as e:
        logger.warning(f"DistilBERT preload failed: {e}")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Text Risk Classifier",
    description=(
        "Classifies social media text into three mental health risk tiers: "
        "Low Risk (0), Moderate Risk (1), High Risk (2). "
        "Supports two models: TF-IDF + Logistic Regression (fast, lightweight) "
        "and DistilBERT (fine-tuned transformer, higher accuracy)."
    ),
    version="1.0.0",
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
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Social media post text to classify.",
        examples=["I have been feeling really low lately and can't sleep or focus on anything."],
    )


class PredictResponse(BaseModel):
    risk_tier:     int   = Field(..., description="Predicted risk tier: 0=Low, 1=Moderate, 2=High")
    risk_label:    str   = Field(..., description="Human-readable risk label")
    confidence:    float = Field(..., description="Model confidence for predicted class (0–1)")
    probabilities: dict  = Field(..., description="Softmax probabilities for all three classes")
    model:         str   = Field(..., description="Model used: 'tfidf' or 'distilbert'")
    latency_ms:    float = Field(..., description="Server-side inference time in milliseconds")


class HealthResponse(BaseModel):
    status:       str
    models_loaded: list[str]
    uptime_s:     float
    version:      str


# ── State ─────────────────────────────────────────────────────────────────
_start_time = time.time()


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Minimal health check — returns 200 if the service is up."""
    return {"status": "ok", "service": "mental-health-classifier"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Detailed health check showing which models are currently loaded."""
    from model_loader import _models
    return {
        "status":        "ok",
        "models_loaded": list(_models.keys()),
        "uptime_s":      round(time.time() - _start_time, 1),
        "version":       "1.0.0",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(
    request: PredictRequest,
    model: Literal["tfidf", "distilbert"] = Query(
        default="distilbert",
        description="Model to use for inference. 'tfidf' is faster; 'distilbert' is more accurate.",
    ),
):
    """
    Classify a social media post into a mental health risk tier.

    - **model=tfidf** — TF-IDF + Logistic Regression. ~0.4ms inference, 3.2 MB.
      Best for high-throughput or cost-sensitive deployments.
    - **model=distilbert** — Fine-tuned DistilBERT. ~6–15ms inference, 269 MB.
      Best for accuracy-critical applications.

    Returns the predicted risk tier (0/1/2), label, confidence,
    per-class probabilities, model name, and server-side latency.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Request text is empty after stripping whitespace.")

    try:
        if model == "tfidf":
            result = predict_tfidf(text)
        else:
            result = predict_distilbert(text)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model artefacts not found: {e}. "
                   f"Ensure models are available at the configured paths."
        )
    except Exception as e:
        logger.exception(f"Inference error (model={model}): {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    logger.info(
        f"[{model}] '{text[:60]}{'...' if len(text)>60 else ''}' "
        f"→ {result['risk_label']} ({result['confidence']:.2f}) "
        f"in {result['latency_ms']:.2f}ms"
    )

    return result


@app.get("/labels", tags=["Metadata"])
def labels():
    """Return the label mapping for all three risk tiers."""
    return {
        "label_map": {str(k): v for k, v in RISK_NAMES.items()},
        "description": {
            "0": "Low Risk — Normal social media content, no significant distress signals.",
            "1": "Moderate Risk — Anxiety, stress, or personality-related distress.",
            "2": "High Risk — Depression, suicidal ideation, or bipolar-related content.",
        }
    }
"""
locustfile.py — Load testing for Mental Health Risk Classifier
CMPT 756 — Containerized (GKE) vs Serverless (Cloud Run)

Tests 2 platforms × 2 models × 3 traffic patterns = 12 configurations.

Platforms:
  GKE        — containerized, self-orchestrated, always-warm pods
  Cloud Run  — serverless containers, scales to zero, cold starts possible

BEFORE RUNNING — set these environment variables:
  export GKE_URL="http://<GKE_INGRESS_IP>"
  export CLOUDRUN_URL="https://mh-classifier-cloudrun-xxx-uc.a.run.app"

USAGE — run one configuration at a time:

  Scenario 1: Steady traffic (1 req/sec, 60s)
    locust -f tests/locustfile.py GkeTfIdf \
      --headless -u 1 -r 1 --run-time 60s \
      --csv results/gke_tfidf_steady

  Scenario 2: Burst traffic (0->100 users over 10s, hold 90s)
    locust -f tests/locustfile.py GkeTfIdf \
      --headless -u 100 -r 10 --run-time 100s \
      --csv results/gke_tfidf_burst

  Scenario 3: Cold start (1 user, run AFTER 15+ min idle on Cloud Run)
    locust -f tests/locustfile.py ColdStartProbe \
      --headless -u 1 -r 1 --run-time 30s \
      --csv results/cloudrun_tfidf_coldstart

  Available classes:
    GkeTfIdf | GkeDistilBert | CloudRunTfIdf | CloudRunDistilBert | ColdStartProbe

AUTO-SCALING OBSERVATION:

  GKE (run in a separate terminal during burst test):
    kubectl get hpa -n mh-classifier -w
    kubectl get pods -n mh-classifier -w

  Cloud Run replica count:
    gcloud run services describe mh-classifier-cloudrun \
        --region us-central1 \
        --format="value(status.observedGeneration)"

METRIC DEFINITIONS:
  p50/p95/p99 latency  -- Locust CSV _stats.csv columns 50% / 95% / 99%
  throughput           -- Locust CSV requests/s column
  error rate           -- failure_count / request_count * 100
  cold-start latency   -- p99 of first request after 15 min idle (Cloud Run)
  scale-up latency     -- seconds from burst start until p99 stabilises < 500ms (GKE)
  replica count        -- REPLICAS column from kubectl get hpa -w at 5s intervals
"""

import os
import random
import time
import logging
from locust import HttpUser, task, between, constant, events
from locust.exception import StopUser

logger = logging.getLogger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────
GKE_URL      = os.environ.get("GKE_URL",      "http://localhost:8080")
CLOUDRUN_URL = os.environ.get("CLOUDRUN_URL", "http://localhost:8080")

# ── Test payloads ──────────────────────────────────────────────────────────
# Mix of risk levels and text lengths.
# Short inputs tokenise faster (lower latency baseline).
# Long inputs stress max_length truncation (higher latency variance).
TEST_TEXTS = [
    # Low risk
    "I had a great day today. Went for a walk and felt really good.",
    "Just finished a book I really enjoyed. Feeling relaxed.",
    "Cooking dinner and listening to music. Life is good.",

    # Moderate risk
    "Work has been really stressful lately. I cannot seem to catch a break. "
    "Everything feels overwhelming and I do not know how to manage it all.",
    "I have been anxious about my exams for weeks. I keep overthinking everything "
    "and cannot sleep properly. My mind just does not stop.",
    "Feeling really burnt out. I am exhausted all the time and even small tasks "
    "feel impossible. I do not enjoy things the way I used to.",

    # High risk
    "I have been having thoughts about not wanting to be here anymore. "
    "Everything feels hopeless and I do not see a way out. I am so tired.",
    "I cannot stop crying and I do not even know why. I feel completely empty "
    "and disconnected from everyone around me.",
    "The depression has gotten so bad that I cannot get out of bed. "
    "I have not eaten properly in days. I feel like a burden to everyone "
    "and I keep thinking that things would be better without me.",

    # Edge cases
    "sad",
    "I hate myself",
    "fine I guess",
    "everything is great",
]


def _payload():
    return {"text": random.choice(TEST_TEXTS)}


# ── Base class ─────────────────────────────────────────────────────────────
class BasePredictor(HttpUser):
    abstract = True

    def _predict(self, endpoint: str, name: str):
        with self.client.post(
            endpoint,
            json=_payload(),
            name=name,
            catch_response=True,
            timeout=120,
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    latency = data.get("latency_ms", -1)
                    if latency > 500:
                        logger.warning(f"High latency: {latency}ms | {name}")
                    response.success()
                except Exception:
                    response.failure("Response not valid JSON")
            elif response.status_code == 429:
                response.failure("Rate limited (429)")
            else:
                response.failure(f"HTTP {response.status_code}")


# ── GKE — containerized ────────────────────────────────────────────────────
class GkeTfIdf(BasePredictor):
    """
    GKE TF-IDF pod.
    Ingress routes /tfidf/predict -> tfidf-service -> tfidf pods.
    HPA: min=2 max=10, CPU target 60%.
    readinessProbe initialDelay=15s.
    """
    host = GKE_URL
    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        self._predict("/tfidf/predict", "[gke] tfidf")


class GkeDistilBert(BasePredictor):
    """
    GKE DistilBERT pod.
    Ingress routes /distilbert/predict -> distilbert-service -> distilbert pods.
    HPA: min=1 max=4, CPU target 60%.
    readinessProbe initialDelay=45s -- scale-up is slower than TF-IDF.
    This asymmetry is a key finding: large model pods take longer to become
    ready, so GKE scale-up latency is model-size dependent.
    """
    host = GKE_URL
    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        self._predict("/distilbert/predict", "[gke] distilbert")


# ── Cloud Run — serverless containers ─────────────────────────────────────
class CloudRunTfIdf(BasePredictor):
    """
    Cloud Run TF-IDF.
    min-instances=0: cold starts occur after idle period.
    TF-IDF cold start: ~170ms (joblib load, no torch import).
    """
    host = CLOUDRUN_URL
    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        self._predict("/predict?model=tfidf", "[cloudrun] tfidf")


class CloudRunDistilBert(BasePredictor):
    """
    Cloud Run DistilBERT.
    Cold start: ~1000ms+ (torch + transformers import + 269MB model load).
    This is hypothesis H1: DistilBERT cold start is 5-10x worse than TF-IDF
    on serverless, because model size directly scales cold-start cost.
    On GKE, both models are always warm -- no such penalty.
    """
    host = CLOUDRUN_URL
    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        self._predict("/predict?model=distilbert", "[cloudrun] distilbert")


# ── Cold-start isolation probe (Cloud Run only) ────────────────────────────
class ColdStartProbe(HttpUser):
    """
    Single-shot cold-start measurement.
    Run this AFTER a 15+ minute idle period on Cloud Run.
    Cloud Run recycles instances after ~10 min idle by default.

    Sends exactly one request and stops.
    Wall time = cold-start overhead + model inference time.
    model_latency_ms is returned in the response body (measured server-side).
    cold_overhead_ms = wall_ms - model_latency_ms

    To test TF-IDF cold start:    keep endpoint as /predict?model=tfidf
    To test DistilBERT cold start: change to /predict?model=distilbert

    Usage:
      locust -f tests/locustfile.py ColdStartProbe \
        --headless -u 1 -r 1 --run-time 30s \
        --csv results/cloudrun_tfidf_coldstart
    """
    host = CLOUDRUN_URL
    wait_time = constant(0)
    _sent = False

    # Change model here for DistilBERT cold start measurement
    ENDPOINT = "/predict?model=tfidf"

    @task
    def probe(self):
        if self._sent:
            raise StopUser()
        self._sent = True

        wall_start = time.perf_counter()
        with self.client.post(
            self.ENDPOINT,
            json={"text": "I feel completely hopeless and cannot go on."},
            name="[coldstart] probe",
            catch_response=True,
            timeout=60,
        ) as response:
            wall_ms = (time.perf_counter() - wall_start) * 1000
            if response.status_code == 200:
                model_ms = response.json().get("latency_ms", -1)
                overhead_ms = wall_ms - model_ms
                logger.info(
                    f"COLD START RESULT | "
                    f"wall_ms={wall_ms:.0f} | "
                    f"model_ms={model_ms:.1f} | "
                    f"overhead_ms={overhead_ms:.0f}"
                )
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

class ColdStartProbeTfIdf(HttpUser):
    """Cold-start probe for dedicated TF-IDF Cloud Run service."""
    host = os.environ.get("TFIDF_SERVICE_URL", "http://localhost:8080")
    wait_time = constant(0)
    _sent = False
    ENDPOINT = "/predict?model=tfidf"

    @task
    def probe(self):
        if self._sent:
            raise StopUser()
        self._sent = True
        wall_start = time.perf_counter()
        with self.client.post(
            self.ENDPOINT,
            json={"text": "I feel completely hopeless and cannot go on."},
            name="[coldstart-tfidf] probe",
            catch_response=True,
            timeout=120,
        ) as response:
            wall_ms = (time.perf_counter() - wall_start) * 1000
            if response.status_code == 200:
                model_ms = response.json().get("latency_ms", -1)
                overhead_ms = wall_ms - model_ms
                logger.info(
                    f"COLD START TF-IDF | "
                    f"wall_ms={wall_ms:.0f} | "
                    f"model_ms={model_ms:.1f} | "
                    f"overhead_ms={overhead_ms:.0f}"
                )
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class ColdStartProbeDistilBert(HttpUser):
    """Cold-start probe for dedicated DistilBERT Cloud Run service."""
    host = os.environ.get("DISTILBERT_SERVICE_URL", "http://localhost:8080")
    wait_time = constant(0)
    _sent = False
    ENDPOINT = "/predict?model=distilbert"

    @task
    def probe(self):
        if self._sent:
            raise StopUser()
        self._sent = True
        wall_start = time.perf_counter()
        with self.client.post(
            self.ENDPOINT,
            json={"text": "I feel completely hopeless and cannot go on."},
            name="[coldstart-distilbert] probe",
            catch_response=True,
            timeout=300,
        ) as response:
            wall_ms = (time.perf_counter() - wall_start) * 1000
            if response.status_code == 200:
                model_ms = response.json().get("latency_ms", -1)
                overhead_ms = wall_ms - model_ms
                logger.info(
                    f"COLD START DISTILBERT | "
                    f"wall_ms={wall_ms:.0f} | "
                    f"model_ms={model_ms:.1f} | "
                    f"overhead_ms={overhead_ms:.0f}"
                )
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


# ── Event hooks ────────────────────────────────────────────────────────────
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info(
        f"Test started | "
        f"users={environment.runner.target_user_count}"
    )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    t = environment.runner.stats.total
    logger.info(
        f"Test finished | "
        f"requests={t.num_requests} | "
        f"failures={t.num_failures} | "
        f"p50={t.get_response_time_percentile(0.50):.0f}ms | "
        f"p95={t.get_response_time_percentile(0.95):.0f}ms | "
        f"p99={t.get_response_time_percentile(0.99):.0f}ms | "
        f"rps={t.current_rps:.1f}"
    )

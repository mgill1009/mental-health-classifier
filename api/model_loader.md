## `model_loader.py`

Loads and caches both models (TF-IDF and DistilBERT) in memory. Models are loaded once on first use and reused for all subsequent requests — no reloading on every prediction.

### Usage
```python
from model_loader import predict_tfidf, predict_distilbert

result = predict_tfidf("I have not been able to get out of bed for weeks")
result = predict_distilbert("I have not been able to get out of bed for weeks")
```

Both functions return the same structure:
```json
{
  "risk_tier": 2,
  "risk_label": "High Risk",
  "confidence": 0.93,
  "probabilities": {
    "Low Risk": 0.06,
    "Moderate Risk": 0.01,
    "High Risk": 0.93
  },
  "model": "distilbert",
  "latency_ms": 14.2
}
```

To preload both models at startup (avoids cold start on the first request):
```python
from model_loader import preload_all
preload_all()
```

### Notes

- Model paths default to `models/baseline/pipeline.pkl` and `models/distilbert/distilbert-mental-health/`
- Override paths via environment variables: `BASELINE_PATH`, `DISTILBERT_PATH`
- `scikit-learn` must be pinned to `1.8.0` to match the version used to save `pipeline.pkl`
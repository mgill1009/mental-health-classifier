# mental-health-classifier
### Containerized vs. Serverless Deployment on GCP

A 3-class mental health risk classifier comparing two cloud deployment paradigms — **Google Cloud Run** (containerized) vs **Google Cloud Functions** (serverless) — for serving ML inference workloads.

> ⚠️ **Research prototype only.** Not intended for clinical use, diagnosis, or crisis response.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Classification Task](#classification-task)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Notebooks](#notebooks)
- [Deployment](#deployment)
- [Load Testing](#load-testing)
- [Known Limitations](#known-limitations)
- [Team](#team)

---

## Overview

This project investigates the trade-offs between containerized and serverless deployment for ML inference under varying traffic conditions.

**Core research question:** Under what traffic conditions and model complexity does each deployment model outperform the other on cost, latency, and reliability?

Two models are compared:

| Model | Type | Cold Start | Purpose |
|---|---|---|---|
| TF-IDF + Logistic Regression | Sparse bag-of-words + linear classifier | ~220ms | Lightweight deployment baseline |
| DistilBERT (fine-tuned) | Transformer, 66M parameters | ~1,039ms | High-accuracy production model |

Both models are served through the same FastAPI endpoint, deployed on both Cloud Run and Cloud Functions, and benchmarked with Locust under identical traffic scenarios.

---

## Classification Task

**Input:** A social media post (text string)  
**Output:** One of three mental health risk tiers

| Label | Risk Tier | Original Labels | Training Rows |
|---|---|---|---|
| 0 | 🟢 Low Risk | Normal | 15,569 (29.9%) |
| 1 | 🟡 Moderate Risk | Anxiety · Stress · Personality Disorder | 8,491 (16.3%) |
| 2 | 🔴 High Risk | Depression · Suicidal · Bipolar | 28,003 (53.8%) |

---

## Results

### Model Performance (Test Set)

| Metric | TF-IDF + LogReg | DistilBERT | Δ |
|---|---|---|---|
| Accuracy | 0.7903 | **0.8357** | +0.045 |
| Macro-F1 | 0.7837 | **0.8262** | +0.043 |
| Weighted-F1 | 0.7923 | **0.8363** | +0.044 |
| High Risk Recall | 0.77 | **0.86** | +0.09 |
| High Risk FN → Low Risk | 46 (9.3%) | **14 (2.8%)** | −70% |

### Deployment Latency

| Metric | TF-IDF + LogReg | DistilBERT | Ratio |
|---|---|---|---|
| Model size | 3.2 MB | 269 MB | 84× |
| Load time p99 *(cold start)* | 220 ms | 1,039 ms | 4.7× |
| Predict p50 | 0.37 ms | 6.0 ms | 16× |
| Predict p99 | 0.72 ms | 15.1 ms | 21× |

> DistilBERT latency measured on Colab T4 GPU. Cloud Functions (CPU) expected ~150–300ms p99.

### Hypotheses Status

| ID | Hypothesis | Status |
|---|---|---|
| H1 | DistilBERT cold start 5–10× longer than TF-IDF on serverless | ✅ Confirmed (4.7×) |
| H2 | Serverless cheaper for <500 requests/day | ⏳ Pending load tests |
| H3 | p99 tail latency 3× higher on serverless under burst traffic | ⏳ Pending load tests |
| H4 | INT8 quantisation reduces DistilBERT cold start by >40% | ⏳ Pending |

---

## Project Structure

---

## Setup & Usage


---
## Notebooks

### `01_eda_preprocessing.ipynb`
- Loads and merges primary + secondary datasets (~102k raw → 61k after dedup)
- Identifies 79.6% text overlap between sources (shared upstream Kaggle corpus)
- Maps 7 original labels → 3 risk tiers
- Two-mode text cleaning pipeline (`text_bert` / `text_tfidf`)
- Saves processed splits to `data/processed/`

### `02_baseline_model.ipynb`
- TF-IDF (50k features, 1–2gram, sublinear TF) + Logistic Regression
- Test macro-F1: **0.784** | High Risk recall: **0.77**
- 5-fold CV: 0.8532 ± 0.0031 (stable)
- Predict p99: **0.72ms** | Model size: **3.2 MB**
- Saves `pipeline.pkl` to `models/baseline/`

### `03_distilbert_finetune.ipynb` *(Google Colab GPU)*
- Fine-tunes `distilbert-base-uncased` with weighted cross-entropy loss
- Best epoch: 3 of 5 (early stopping on val macro-F1)
- Test macro-F1: **0.826** | High Risk recall: **0.86**
- High Risk FN→Low Risk reduced by **70%** (46 → 14)
- Saves model to `models/distilbert/` and pushes to HuggingFace Hub

** Model on HuggingFace:** [`mgill7436/distilbert-mental-health-risk`](https://huggingface.co/mgill7436/distilbert-mental-health-risk)

---

## Deployment

### Cloud Run (Containerized)

### Cloud Functions (Serverless)

## Load Testing

Three Locust scenarios across all 4 deployment configurations (2 models × 2 platforms):
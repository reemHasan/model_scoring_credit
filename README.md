---
license: mit
title: Loan Scoring App
sdk: docker
emoji: 📊
colorFrom: purple
colorTo: pink
---
<p align="center">
  <img src="home_credit_scoring_logo.png" width="300">
  <br>
</p>

# Home Credit Loan Scoring — MLOps Project

> End-to-end MLOps pipeline for a LightGBM loan default classifier:
> model serving, monitoring, drift detection, profiling, and ONNX optimisation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Model & Data](#3-model--data)
4. [API FastAPI + Gradio](#4-api--fastapi--gradio)
5. [Logging & Storage](#5-logging--storage)
6. [Testing](#6-testing)
7. [Monitoring Dashboard](#7-monitoring-dashboard)
8. [Data Drift Detection](#8-data-drift-detection)
9. [Profiling & Optimisation](#9-profiling--optimisation)
10. [Containerisation](#10-containerisation)
11. [CI/CD](#11-cicd)
12. [Deployment — HuggingFace Spaces](#12-deployment--huggingface-spaces)
13. [How to Run Locally](#13-how-to-run-locally)

---

## 1. Project Overview

This project implements a full MLOps pipeline around a **LightGBM binary classifier**
trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
dataset. Given a client ID, the model predicts whether a loan applicant will default,
and returns SHAP explanations for the decision.

### Architecture

```
User
 │
 ▼
Gradio UI (HuggingFace Space 1)
 │
 ▼
FastAPI /predict endpoint
 ├── LightGBM / ONNX inference
 ├── SHAP explanation
 ├── JSON logger  (stdout + file)
 └── PostgreSQL store (BackgroundTask)
         │
         ▼
    Supabase PostgreSQL
         │
         ▼
Streamlit Dashboard (HuggingFace Space 2)
 ├── KPI metrics
 ├── Latency charts
 ├── Score & class drift
 └── Evidently AI reports
```

---

## 2. Project Structure

```
model_scoring_credit/
│
├── app/                          # API source code
│   ├── api.py                    # FastAPI app + lifespan
│   ├── gui.py                    # Gradio interface (mounted inside FastAPI)
│   ├── predict_service.py        # predict_and_log() — inference + logging
│   ├── logger.py                 # Structured JSON logger (stdout + file)
│   ├── database.py               # SQLAlchemy ORM + PostgreSQL storage
│   ├── state_store.py            # Shared state between FastAPI and Gradio
│   └── profiling/
│       └── profiler.py           # PyInstrument middleware
│
├── monitoring_api/                    # Streamlit monitoring dashboard
│    ├── dashboard.py
|    ├── db_helpers.py
|    ├── Dockerfile
|    ├── README.md
│    └── requirements.txt
│
├── ml/                           # Model artifacts
│   └── model/
│       ├── lgbm_bestmodel_fbeta10_bundle.pkl
│       ├── lgbm_model.onnx
│       └── lgbm_model_quantized.onnx
|        └── src/                      # One-off utility scripts
│           ├── convert_to_onnx.py
│           ├── quantize_onnx.py
│           └──benchmark.py 
│
├── data/prod_data/               # Production data
│   ├── new_test_data_20features.parquet
│   ├── shap_values.parquet
│   └── expected_value.pkl
│
├── test/                        # Pytest test suite
│   ├── conftest.py
│   └── test_api.py
│
├── .github/workflows/
│   ├── ci.yml                    # Tests on every PR
│   ├── cd_api.yml                # Deploy API to HF Space
│   └── cd_dashboard.yml          # Deploy dashboard to HF Space
│
├── Dockerfile                    # API container (HF Spaces compatible)
├── docker-compose.yml  
├── pyproject.toml                # Dependencies managed by uv
├── uv.lock
└── README.md
```

---

## 3. Model & Data

### Data
  The Home Credit dataset consists of 307,511 training instances and 48,744 test instances, each described by 122 features. The original application data were enriched through the integration and aggregation of information from historical and external data sources.

### Model
- **Algorithm**: LightGBM binary classifier
- **Objective**: predict loan default (class 1 = default)
- **Threshold**: custom F-beta threshold optimised for recall
- **Features**: 20 engineered features from the Home Credit dataset
- **Explainability**: SHAP TreeExplainer values pre-computed for all test clients

---

## 4. API — FastAPI + Gradio

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check — returns model loaded status |
| `POST` | `/predict/{loan_id}` | Predict default probability for client ID |
| `GET`  | `/` | Gradio UI (mounted at root) |

### Predict response
```json
{
  "request_id": "abc12345",
  "Client_id": 42,
  "Client default probability": 0.73,
  "Class": "default",
  "Decision": "Reject loan application",
  "inference_ms": 12.4,
  "total_ms": 18.1,
  "Client_info": "[{...}]",
  "Expected_Shap_Value": 0.12,
  "Shap_values_client": "[{...}]",
  "model_runtime": "lightgbm"
}
```

### Key design decisions

**Gradio mounted inside FastAPI** — single process, single port (7860).
No separate Gradio server needed:
```python
from gradio.routes import mount_gradio_app
app = mount_gradio_app(app, demo, path="/")
```

**`app.state` via `state_store.py`** — solves the Gradio wrapper issue where
`mount_gradio_app` creates a new app object. Both FastAPI endpoints and Gradio
functions access state via `get_state()` / `request.app.state`:
```python
# api.py
result = run_prediction(loan_id, request.app.state)

# gui.py
app_state = get_state()
result    = await predict_and_log(loan_id, app_state)
```

**`predict_and_log()` in `predict_service.py`** — all prediction logic,
logging, and DB scheduling separated from the HTTP layer:
```python
async def predict_and_log(loan_id, state, background_tasks=None):
    result = run_prediction(loan_id, state)
    log_prediction(...)                    # fast — sync
    _schedule(store_record, ...)           # slow — background
    return result
```

---

## 5. Logging & Storage

### Structured JSON logging (`logger.py`)

Every API call emits one JSON line to stdout and `logs/api.log`:
```json
{
  "ts": "2026-05-14T10:23:01Z",
  "level": "INFO",
  "event": "prediction",
  "request_id": "abc123",
  "loan_id": 42,
  "proba_default": 0.73,
  "inference_ms": 12.4,
  "total_ms": 18.1,
  "status_code": 200,
  "model_runtime": "lightgbm"
}
```

File logging is disabled automatically on HuggingFace Spaces
(detected via `SPACE_ID` env var) since the disk is ephemeral.

### PostgreSQL storage (`database.py`)

**Stack**: SQLAlchemy async ORM + psycopg v3

**Schema** — single table, one row per prediction:
```python
class LoanApiLog(Base):
    __tablename__ = "loan_api_logs"
    request_id:    str
    timestamp:     datetime
    loan_id:       int
    proba_default: float
    proba_class:   str       # "default" / "no default"
    decision:      str
    inference_ms:  float
    total_ms:      float
    status_code:   int
    error_message: str
    features:      JSON      # all 20 input features in one row
    shap_values:   JSON      # SHAP values dict
    model_runtime: str
```

Features stored as `JSON` column (not EAV rows) — one prediction = one row.
`pd.json_normalize(df["features"])` expands to wide format for Evidently.

**BackgroundTasks pattern** — DB write never blocks the API response:
```
request → inference → log → return response
                               ↓ (after response sent)
                          store_record() → PostgreSQL
```

When called from Gradio (no BackgroundTasks object available),
`asyncio.create_task()` is used instead — same fire-and-forget behavior.

---
## 6. Testing

**Stack**: pytest + unittest.mock

### Test classes

| Class | What it tests |
|-------|---------------|
| `TestHealthEndpoint` | GET /health — status, model loaded |
| `TestPredictEndpoint` | POST /predict — input validation, response structure, decision logic |
| `TestLogging` | `log_prediction()` called with correct args per request |
| `TestDBStore` | `store_record()` called correctly — including BackgroundTask flush |
| `TestPredictor` | `run_prediction()` unit tests — no HTTP overhead |

### Key mocking strategy

```python
# All external dependencies mocked — no real files, no real DB
patch("app.api.init_db",              new_callable=AsyncMock)   # no DB at startup
patch("gradio.routes.mount_gradio_app", ...)                     # no Gradio UI
patch("app.logger.log_prediction")                               # no file writes
patch("app.database.store_record",    new_callable=AsyncMock)   # no DB calls
```

Patch targets are at **definition site** (`app.logger`, `app.database`),
not usage site — required because modules are imported before patches activate.
<!---
### BackgroundTasks in tests

`store_record` runs as a `BackgroundTask` — the `TestClient` context manager
must close before assertions to flush pending tasks:
```python
with TestClient(app) as c:
    c.post("/predict/1")
# ← client closed here — background tasks flushed
assert mock_store.called   # ← now safe
```

### Run tests
```bash
pytest test/test_api.py -v --tb=short --cov=app --cov-report=term-missing
```-->

---

## 7. Monitoring Dashboard

**Stack**: Streamlit + Plotly + Evidently AI

### Panels

| Panel | Description |
|-------|-------------|
| Latency Comparison Lightgbm & ONNX  | latency box plot on-demand via button|
| Inference Comparison Lightgbm & ONNX | Mean inference vs P95 bar chart on-demand via button|
| KPI row | Total calls, error rate, max latency, avg latency, avg inference, rejection rate |
| Latency over time | `total_ms` line chart with anomaly threshold (mean + 2σ) |
| Inference over time | `inference_ms` line chart |
| Error distribution | Status code bar chart — only shown when error rate > 0 |
| Score over time | Avg predicted probability per class over time |
| Class drift | Reference vs current default rate — grouped bar chart |
| Evidently DataDrift | Full HTML report embedded via `st.components.v1.html` |
| Evidently DataSummary | On-demand via radio button |


### Auto-refresh
Configurable from sidebar (1–24 hours) via `streamlit-autorefresh`.
Evidently reports are cached for 3 hours (`@st.cache_data(ttl="3h")`)
to avoid re-generating on every refresh.

---

## 8. Data Drift Detection

**Library**: Evidently AI

### Reference vs current split
The sidebar slider controls what % of earliest data is used as the
reference baseline (default: 30%). The rest is the current window.

### What is monitored

| Signal | Method | Alert |
|--------|--------|-------|
| Feature drift | KS test / chi² per feature | Evidently drift flag |
| Class drift | Default rate reference vs current | > 10 percentage points |
| Error rate | status_code ≥ 400 / total | > 5% |
---

## 9. Profiling & Optimisation

### Step 1 — Profile with PyInstrument

PyInstrument was chosen over cProfile because:
- Native async/await support — shows real coroutine time
<!--- No `ProactorEventLoop` conflict on Windows -->
- Clean flame-tree output vs flat function list

### Profiling findings (3 requests)

| Request | Bottleneck | % time | Root cause |
|---------|-----------|--------|------------|
| 1st | `LGBMClassifier.predict_proba` | 88% | Model cold start |
| 2nd | `AsyncSession.commit` | 43% | DB connection cold start |
| 3rd+ | `AsyncSession.commit` | 60% | DB write blocking response |

### Step 2 — Optimisations applied

**Fix 1 — Model pre-warming** (eliminates cold start on request 1):
```python
# lifespan — run one dummy prediction at startup
session.run([out_name], {input_name: dummy_X})
```

**Fix 2 — BackgroundTasks** (eliminates DB bottleneck on all requests):
```python
# DB write moved after response — user never waits for it
background_tasks.add_task(store_record, ...)
return result   # ← user gets response here, DB write happens after
```

### Step 3 — ONNX conversion + quantization
Performance profiling identified model inference as a potential optimization target. The LightGBM model was converted to ONNX format and executed with ONNX Runtime using the CPUExecutionProvider. To obtain stable measurements, the models were warmed up before benchmarking and each configuration was evaluated over 1000 consecutive predictions.

The ONNX float32 implementation reduced inference latency by approximately 20.97× compared with the original LightGBM model. Static INT8 quantization provided an overall speedup of 23.31× relative to the initial implementation.

Since quantization preserved prediction accuracy while slightly improving latency, the quantized INT8 model was selected as the final production model.


```bash
# Convert LightGBM → ONNX float32
python ml/src/convert_to_onnx.py

# Quantize ONNX float32 → INT8
python ml/src/quantize_onnx.py

# Benchmark all 3 versions
python ml/src/benchmark.py
```

### Benchmark results (typical)

| Version | Mean latency | Speedup |
|---------|-------------|---------|
| LightGBM (baseline) | 1.223296 | 1x |
| ONNX float32 | 0.058343 | 20.97× faster |
| ONNX INT8 quantized | 0.052482 | 23.31× faster |
| INT8 vs ONNX FP32 |  -- | 1.11× faster |

---

## 10. Containerisation

### Local development
```bash
# Start API + Gradio UI
docker compose -f docker-compose.yml up --build
```

### Multi-stage Dockerfile

```
Stage 1 (builder):  python:3.12-slim + uv → installs .venv
Stage 2 (runtime):  python:3.12-slim + libgomp1 + .venv only
```

Size reduction: ~1.1GB → ~620MB by excluding uv, gcc, and build tools.

`libgomp1` is installed in the runtime stage — required by LightGBM,
numba, and scikit-learn for OpenMP parallel computation.

---

## 11. CI/CD

### Workflows

| File | Trigger | Does |
|------|---------|------|
| `ci.yml` | Every PR + push to main | Runs pytest |
| `cd_api.yml` | CI passes + `app/` changed | Deploys API to HF Space |
| `cd_dashboard.yml` | `monitoring_api/` changed | Deploys dashboard to HF Space |

### Smart path filtering

CD workflows only deploy when relevant files change:
```yaml
# cd_api.yml — only triggers when API files change
paths-ignore:
  - "monitoring_api/**"

# cd_dashboard.yml — only triggers when dashboard files change
paths:
  - "monitoring_api/**"
```

### Required GitHub secrets

| Secret | Used by |
|--------|---------|
| `HF_TOKEN` | cd_api.yml, cd_dashboard.yml |
| `HF_SPACE_ID` | cd_api.yml |
| `HF_DASHBOARD_SPACE_ID` | cd_dashboard.yml |

---

## 12. Deployment — HuggingFace Spaces

### Two separate Spaces

| Space | Stack | URL |
|-------|-------|-----|
| Space 1 — API | FastAPI + Gradio + LightGBM/ONNX | [Loan credit scoring api](https://reemmathbout-loan-scoring-app.hf.space) |
| Space 2 — Dashboard | Streamlit | [Monitoring Loan credit scoring api](https://reemmathbout-monitoring-loan-scoring-api.hf.space) |

Both Spaces connect to the same **Supabase PostgreSQL** database via secrets.

---

## 13. How to Run Locally

### Prerequisites
```bash
# Python 3.12
python --version

# uv package manager
pip install uv

# Docker Desktop (for containerised run)
docker --version
```

### Install dependencies
```bash
uv sync
```

### Run API directly
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000
```

### Run with Docker Compose
```bash
docker compose -f docker-compose.yml up --build
# → http://localhost:8000
```

### Run Streamlit dashboard
```bash
streamlit run monitoring_api/dashboard.py --server.runOnSave true
# → http://localhost:8501
```

### Run tests
```bash
pytest
```

### Run profiling
```bash
$env:ENABLE_PROFILING="true"; uvicorn app.api:app --port 8000
1..20 | ForEach-Object {Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/predict/$_"}

# opens profiling_output/profiles/predict_N.html in browser
```

### Convert and benchmark ONNX
```bash
python scripts/convert_to_onnx.py
python scripts/quantize_onnx.py
python scripts/benchmark.py
```

---

## Stack Summary

| Layer | Technology |
|-------|-----------|
| ML model | LightGBM + ONNX Runtime |
| API | FastAPI + Uvicorn |
| UI | Gradio |
| Logging | Python logging + JSON formatter |
| Storage | PostgreSQL (Supabase) + SQLAlchemy async ORM |
| Monitoring | Streamlit + Plotly |
| Drift detection | Evidently AI |
| Profiling | PyInstrument |
| Testing | Pytest + unittest.mock |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | HuggingFace Spaces |
| Package manager | uv |


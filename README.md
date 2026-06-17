# Network Anomaly Detection System (CICIDS2017)

A deployable, fully-tested machine learning system for real-time network traffic anomaly detection. It learns normal network traffic behaviour from benign flow data (unsupervised) and flags deviations as potential security incidents.

## 🏗️ Architecture Overview

The system runs locally using a multi-service Docker configuration:

- **TimescaleDB** (PostgreSQL 16): Persistent storage for network flows, predictions, alerts, and model metadata.
- **FastAPI API** (Python 3.12): High-performance backend orchestrating database calls, live server-sent events (SSE) feeds, and scoring.
- **Traffic Simulator** (Python 3.12): Replays benchmark flow data by streaming HTTP POST requests to the API at a configurable rate, overriding timestamps to simulate real-time traffic.
- **React Dashboard** (Vite + TS + Recharts): A high-fidelity, real-time glassmorphism UI showing KPI counters, a live anomaly timeline, flow streams, and alert drill-downs.

---

## 📈 Evaluation & Benchmarks

The models were evaluated against a test partition of the CICIDS2017 dataset. PR-AUC is our lead metric due to the severe class imbalance (~20% anomalies in the test set).

### Model Comparison

| Model | Type | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | FPR@90%R |
|-------|------|--------|---------|-----------|--------|-----|-----|---------|
| isolation_forest | unsupervised | 0.4315 | 0.7140 | 0.6000 | 0.0458 | 0.0851 | 0.0085 | 0.6588 |
| **autoencoder** | **unsupervised** | **0.8718** | **0.9149** | **0.9694** | **0.7252** | **0.8297** | **0.0064** | **0.3198** |
| halfspace_trees | unsupervised | 0.3259 | 0.6758 | 0.2000 | 0.0076 | 0.0147 | 0.0085 | 0.6546 |
| lightgbm_benchmark ⚠️ | supervised | 0.9601 | 0.9746 | 0.9833 | 0.9008 | 0.9402 | 0.0043 | 0.0043 |

> ⚠️ = Supervised upper-bound benchmark (uses labels unavailable in production).
> All unsupervised models trained on benign traffic only.
> Threshold selected at target FPR ≤ 1%.

---

## 🚀 Setup & Execution

### Prerequisites
- Docker & Docker Compose
- (Optional, for host dev) Python 3.12+ and `uv`

### Running the Stack (Docker)
1. Initialize the environment variables:
   ```bash
   cp .env.example .env
   ```
2. Start the services:
   ```bash
   docker-compose up --build
   ```
This boots TimescaleDB, performs database migrations, loads the trained model registry, and starts the simulator + frontend dashboard.

### Local Development / Testing on the Host
If you want to run the python test suite and models locally on your host machine:

1. Install `uv` and create a virtual environment in `backend/`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   cd backend
   uv venv --python 3.12
   uv pip install -e ".[dev]" torch tqdm aiosqlite
   ```
2. Setup the local `.env` configuration file:
   ```bash
   # Create backend/.env using local paths
   MODEL_REGISTRY_PATH=../models
   DATA_DIR=../data
   ```
3. Run the complete test suite:
   ```bash
   # On macOS, pass DYLD_LIBRARY_PATH if using LightGBM
   DYLD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python -m pytest -v
   ```

---

## 🛠️ Production Realism Additions

The system includes key enterprise-ready features for operational robustness:

1. **Session-based Authentication:** Gated endpoints require cookies authenticated via the security console. Seeded analyst account: `analyst` / `password123`.
2. **On-Demand Attack Scenario Injector:** Injects DDoS, Port Scan, or Brute Force traffic dynamically into the streaming simulation to validate detection rules.
3. **Data Drift Monitoring:** Tracks feature-level and overall Population Stability Index (PSI) against training deciles. Detects distribution changes and flags a `stale` model.
4. **Analyst Feedback Loop:** Allows marking predictions as True Positives or False Positives. Feedbacks are logged and can be exported as a clean training-ready CSV (`/api/v1/alerts/feedback/export`).
5. **Observability & Health Checks:** Exposed endpoints `/health` (liveness), `/ready` (dependency and model checks), and `/metrics` (Prometheus exposition format tracking request throughput, alert rates, and latency).

---

## ⚠️ Limitations & Notes
1. **FPR/Recall Tradeoff:** The AutoEncoder shows excellent performance but online deployment requires careful operational tuning to keep alerts manageable.
2. **Concept Drift:** Network traffic patterns evolve over time. Use the built-in PSI monitor to detect drift and trigger model retraining.


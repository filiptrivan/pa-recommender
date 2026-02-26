# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production ALS (Alternating Least Squares) implicit feedback recommender system for an e-commerce platform. Uses the `implicit` library to generate homepage, similar product, and cross-sell recommendations. Trained offline on a schedule, with results cached in Redis.

## Development Commands

```bash
# Install dependencies (Python 3.11 required)
pip install -r requirements.txt

# Run locally
python app.py                    # Flask dev server on port 8080

# Run with gunicorn (production)
gunicorn -b :8080 app:app --workers 1 --timeout 3600

# Start local MongoDB + Mongo Express
docker-compose up -d             # MongoDB on :27017, Mongo Express on :8081

# Build Docker image (CUDA 11.8 base)
docker build -t pa-recommender .

# Deploy via Google Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

There is no automated test suite. Testing is done via the `/test_hyperparameter_optimization` endpoint which runs ALS parameter grid search with implicit's evaluation metrics (precision@k, AUC@k, MAP@k, NDCG@k). Jupyter notebooks in `examples/` are used for manual analysis.

## Architecture

### Entry Point & API (`app.py`)
Flask app with rate limiting (200/hr). All training endpoints require API key auth via `@shared.require_api_key` decorator. Global exception handler emails tracebacks.

**Endpoints:**
- `GET /train_homepage_and_similar_products_model_by_http_request` — trains ALS model, caches per-user homepage recs and per-product similar items in Redis
- `GET /train_cross_sell_model_by_http_request` — builds session-based product-product similarity matrix, caches in separate Redis instance
- `GET /test_hyperparameter_optimization` — grid search for ALS hyperparameters

### Data Pipeline (`utils/shared.py`)
Central module (~700 lines) containing the full preprocessing pipeline:

1. **Fetch**: External ReadyCMS API → raw interactions + products
2. **Cache**: Previous year stored in Azure Blob Storage; only last 24h fetched fresh
3. **Outlier detection**: Removes bot/crawler traffic and high-frequency users (configurable via `outlier_config.py`)
4. **Interaction weighting**: Action-type weights × exponential time decay (`e^(-days/25)`) × repeat decay (0.8 factor)
5. **Threshold filtering**: Minimum interaction counts per user/product
6. **BM25 weighting**: Applied to sparse user-product matrix before ALS training

Key constants defined at module top: `INTERACTION_WEIGHTS`, `INTERACTION_DECAY_SCALE`, `INTERACTION_REPEAT_DECAY_FACTOR`, column name constants (`PRODUCT_COL_NAME`, `USER_COL_NAME`, etc.).

### Recommendation Engine (`utils/als.py`)
- **Homepage/Similar Products**: ALS with factors=800, regularization=0.005, alpha=800.0, iterations=35. Results cached in Redis with 7-day TTL.
- **Cross-Sell**: Session-based (12-hour windows) product co-occurrence similarity matrix. Uses a separate Redis instance.

### Supporting Modules
- `utils/data.py` — Azure Blob Storage read/write for cached interaction data
- `utils/emailing.py` — Gmail SMTP for error alerts and daily training reports
- `utils/mongodb_client.py` — MongoDB connection for analytics/warehousing
- `utils/outlier_detection.py` + `utils/outlier_config.py` — Configurable outlier detection algorithms
- `utils/classes/Settings.py` — Singleton config loader from `.env` (uses `Singleton` metaclass)
- `DTO/ProductDTO.py` — Pydantic model for product data validation
- `exceptions/BusinessException.py` — Custom exception returning 400 status

### Infrastructure
- **Redis**: Two separate Redis Cloud instances (homepage recs + cross-sell recs)
- **Azure Blob Storage**: Historical interaction data cache
- **MongoDB**: Analytics and data warehousing (local via docker-compose)
- **GCP Cloud Run**: Production deployment with NVIDIA L4 GPU, 16GB RAM, 8 CPUs
- **Azure**: Only handles daily scheduling; all compute on GCP

## Key Conventions

- Vectorized NumPy/pandas operations only — no Python loops for data processing
- Sparse matrices (CSR format via scipy) for memory efficiency
- `ENV` setting in `.env` controls Dev vs Prod behavior (e.g., Azure Monitor only in Prod)
- `OPENBLAS_NUM_THREADS=1` set at startup to avoid thread contention
- `StringBuilder` utility used to aggregate log messages for email reports

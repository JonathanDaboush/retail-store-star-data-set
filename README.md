# Retail Store Star Schema Dataset

Production-style portfolio project showing immutable historical retail data replayed through Kafka into an operational star schema with a FastAPI control plane and React manager dashboard.

## Business purpose

Demonstrate how a retail team can replay historical sales in controlled batches, monitor real processing state, and see incremental business metrics update without rebuilding the warehouse from scratch.

## Architecture

`backend/original_data/fact_sales_normalized.csv` (immutable source)
→ FastAPI replay control
→ Kafka topic (`retail-events-kafka`)
→ idempotent Kafka consumer
→ MySQL `fact_sales_normalized` updates
→ analytics + diagnostics API
→ React dashboard

Airflow orchestrates replay execution (`retail_replay_batch` DAG) via the same public API used by the UI.

## Tech stack

- **Backend:** FastAPI, SQLAlchemy, pandas
- **Streaming:** Kafka (kafka-python), Zookeeper
- **Database:** MySQL 8 (star schema + constraints/indexes)
- **Orchestration:** Apache Airflow 2
- **Frontend:** React + Vite
- **ML assets:** persisted `.pkl` artifacts under `backend/models`

## Repository layout

- `backend/main.py`: API, replay state machine, event-bank selectors, diagnostics, Excel preview
- `backend/controller/producer.py`: Kafka producer with delivery confirmation + logging
- `backend/controller/consumer.py`: continuous consumer with validation, dedupe, failure table, manual commits
- `backend/quick_load.py`: dimension bootstrap + fact constraints/indexes
- `airflow/dags/retail_replay.py`: staged orchestration DAG
- `frontend/src/App.jsx`: manager UI
- `frontend/src/api.js`: centralized API client with timeout/error handling

## Prerequisites

- Docker + Docker Compose
- Node.js 20+

## Environment variables

1. Copy `.env.example` to `.env`.
2. Replace every `REPLACE_WITH_A_SECRET` value.
3. Leave the `ML_MODEL_SHA256_*` values as-is unless you intentionally regenerate the checked-in model artifacts.

Used values:

- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_ROOT_PASSWORD`
- `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC`, `KAFKA_CLIENT_ID`
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `AIRFLOW_UID`, `AIRFLOW_USER`, `AIRFLOW_PASSWORD`, `AIRFLOW_EMAIL`
- `CORS_ORIGINS`
- `ML_MODEL_SHA256_FORECAST`, `ML_MODEL_SHA256_CHURN`, `ML_MODEL_SHA256_LTV`, `ML_MODEL_SHA256_DEMAND`, `ML_MODEL_SHA256_RECOMMENDATION` (required for ML inference endpoints)

Never commit `.env`.

## Local startup (end-to-end)

1. Start infrastructure:
   ```bash
   docker compose up -d mysql zookeeper kafka postgres
   ```
2. Initialize star schema dimensions + empty fact table:
   ```bash
   docker compose run --rm loader
   ```
   The loader creates typed MySQL tables before adding keys/indexes so replay-ready `sales_id` uniqueness and foreign keys can be created successfully on a fresh database.
3. Start API + consumer + Airflow:
   ```bash
   docker compose up -d backend consumer airflow-init airflow-webserver airflow-scheduler
   ```
4. Start frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
5. Open dashboard (usually `http://localhost:5173`).

## Replay/event-bank behavior

- Source file (`backend/original_data/fact_sales_normalized.csv`) is treated as immutable.
- Replay state is stored separately in `replay_state`.
- Supported batch selectors:
  - `events` (full event bank)
  - `day` (`YYYY-MM-DD`)
  - `week` (`YYYY-Www`)
  - `store` (`store_sk`)
- Configurable pacing via `batch_size` + `interval_seconds` (for example: 10/sec, 50 per 5 sec, 500/min depending on values).

## Incremental processing and integrity

- Consumer processes only incoming events (no full table rebuild).
- `processed_events` table provides event-level idempotency.
- `fact_sales_normalized` has PK/unique/indexes and foreign keys to dimensions.
- Malformed/failed events are persisted to `failed_events` and counted in diagnostics.

## API highlights

- `GET /dashboard`: KPI snapshot + revenue trend + top products
- `GET /analytics?task=...`: business analytics queries
- `GET /replay`: live replay status (published/consumed/lag/rate)
- `GET /replay/options`: valid day/week/store selectors from event bank
- `POST /replay/start`: start batch replay
- `POST /replay/control`: pause/resume/stop
- `GET /diagnostics`: API/DB/Kafka/replay health + latest batch + failures
- `GET /ml/status`: ML artifact availability
- `GET /ml/report`: live prediction summaries (forecast/churn/ltv/demand)
- `POST /ml/recommendations`: customer-level product recommendations
- `POST /ml/predict/{model}`: direct model inference for supplied feature records
- `POST /uploads/preview`: immutable Excel preview and validation (`.xlsx`, `.xls`)

## Airflow DAG

`retail_replay_batch` steps:
1. Determine next batch
2. Validate batch settings
3. Publish replay batch
4. Wait for consumer catch-up
5. Refresh incremental analytics
6. Validate replay results
7. Generate summary
8. Record completion

Includes retries and explicit failure paths.

## Frontend behavior

Dashboard provides:
- Revenue/orders/customers/freshness
- Replay controls (mode, selector, size, speed, pause/resume/stop)
- Processing metrics (published/consumed/remaining/lag/rate/failures)
- System health statuses (not color-only)
- Analytics actions and top products
- Excel validation summary

## ML status

Repository includes persisted model artifacts (`backend/models/*.pkl`).
The API exposes artifact status, prediction summaries, direct inference, and recommendation endpoints.

## Verification commands

From repo root:

```bash
python -m compileall -q backend
```

From `frontend`:

```bash
npm run build
```

Optional smoke checks after services are running:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/replay
curl http://localhost:8001/diagnostics
```

## Troubleshooting

- **Replay start returns 409:** an existing replay is still running; pause/stop first.
- **Replay start returns 422:** selected day/week/store has zero matching events.
- **Kafka marked failed in diagnostics:** verify `kafka` container health and `KAFKA_BOOTSTRAP_SERVERS`.
- **Consumer lag grows:** inspect `consumer` logs for DB/FK/validation errors.
- **Airflow DAG fails:** inspect task logs under Airflow UI and API connectivity to `backend:8000`.

## Project status

Implemented: immutable replay, configurable batch selectors, Kafka producer/consumer reliability, incremental DB updates, Airflow orchestration, manager dashboard controls, diagnostics, Excel preview validation, and manager-facing ML prediction/recommendation endpoints.

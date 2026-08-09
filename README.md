# Retail Store Star Schema Dataset

An end-to-end retail operations demonstration: an immutable historical sales CSV is released in paced batches to Kafka, ingested idempotently into a MySQL star schema, and exposed through a manager-friendly React dashboard.

## Architecture

`original_data/fact_sales_normalized.csv` (read-only event bank) → FastAPI replay control → Kafka → consumer → MySQL fact table → dashboard API → React UI. Dimensions load before any facts, preserving referential availability. Airflow services are included for orchestration infrastructure.

## Start locally

1. Copy `.env.example` to `.env` and replace every `REPLACE_WITH_A_SECRET` value. Do not commit this file.
2. Run `docker compose up -d mysql zookeeper kafka`.
3. Run `docker compose run --rm loader` once. This loads dimensions only and creates an empty indexed fact table.
4. Run `docker compose up -d backend consumer airflow-init airflow-webserver airflow-scheduler`.
5. In `frontend`, run `npm install` and `npm run dev`; open the shown Vite URL (normally `http://localhost:5173`).

The API is available at `http://localhost:8001/docs`. The dashboard uses it automatically; set `VITE_API_URL` only when hosting the API elsewhere.

## Manager workflow

Choose the batch size and pause between batches, then select **Start processing**. The dashboard polls the actual API state every four seconds. Kafka publishing and database consumption are separate; the consumer commits offsets only after the MySQL transaction succeeds. The fact-table primary key makes duplicate delivery safe.

## Excel preview

The UI accepts `.xlsx` and `.xls` workbooks up to 25 MB. A content-addressed copy is stored under `backend/data/uploads/original`; previews use that copy and never alter the uploaded workbook. Uploads are ignored by Git.

## Analytics and ML pipeline

The dashboard’s analysis controls query the processed fact table for category revenue, daily revenue, and top customers. The Python pipeline uses the bundled source data by default: `run_initial_pipeline(max_fact_rows=5000)` creates temporal labels from an 80/20 date cutoff, and fits transforms on training data only. Set `train_models=False` to validate/prepare data without model dependencies; model training requires the backend requirements.

## Verification

Run `python -m compileall -q backend` and `npm run build` in `frontend`. For a full run, start the services, run one replay batch, then confirm orders/revenue increase in the dashboard and `GET /diagnostics` reports processed transactions.

## Airflow

The `retail_replay_batch` DAG is manually triggered and starts a 100-event replay batch through the backend API, then validates the recorded state. It retries transient API failures and deliberately reuses the same Kafka publishing logic as the UI.

## Limitations

The existing saved ML artifacts are preserved but do not yet have prediction endpoints or a manager-facing training workflow. They are not represented as working UI functionality.

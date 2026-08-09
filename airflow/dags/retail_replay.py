"""Orchestrate one replay cycle through backend + Kafka + consumer + incremental analytics."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from airflow import DAG
from airflow.operators.python import PythonOperator

API = "http://backend:8000"
DEFAULT_BATCH_SIZE = 100
DEFAULT_INTERVAL_SECONDS = 5


def call(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(f"{API}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} on {path}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network failure on {path}: {exc}") from exc


def determine_next_batch(**context) -> dict:
    replay = call("/replay")
    if replay.get("status") == "running":
        raise RuntimeError("Replay is already running; cancel or wait before triggering DAG.")
    batch_payload = {
        "batch_size": DEFAULT_BATCH_SIZE,
        "interval_seconds": DEFAULT_INTERVAL_SECONDS,
        "batch_mode": "events",
        "batch_value": None,
    }
    context["ti"].xcom_push(key="batch_payload", value=batch_payload)
    return batch_payload


def validate_batch_selection(**context) -> dict:
    payload = context["ti"].xcom_pull(task_ids="determine_next_batch", key="batch_payload")
    options = call("/replay/options")
    total = int(options.get("events_total") or 0)
    if total <= 0:
        raise RuntimeError("Event bank is empty; replay cannot start.")
    if payload["batch_mode"] == "events" and payload["batch_size"] > total:
        payload["batch_size"] = total
    return payload


def publish_batch(**context) -> dict:
    payload = context["ti"].xcom_pull(task_ids="validate_batch", key="return_value")
    response = call("/replay/start", payload)
    if response.get("status") not in {"running", "completed"}:
        raise RuntimeError(f"Replay failed to start: {response}")
    return response


def wait_for_consumer_processing() -> dict:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        replay = call("/replay")
        if replay.get("status") == "failed":
            raise RuntimeError(replay.get("last_error") or "Replay failed")
        if replay.get("events_published", 0) > 0 and replay.get("lag_events", 0) == 0:
            return replay
        time.sleep(5)
    raise TimeoutError("Consumer did not catch up within 10 minutes.")


def refresh_incremental_analytics() -> dict:
    dashboard = call("/dashboard")
    analytics = call("/analytics?task=daily_revenue")
    return {"dashboard": dashboard, "analytics": analytics}


def validate_results(**context) -> dict:
    replay = context["ti"].xcom_pull(task_ids="wait_for_consumer", key="return_value")
    diagnostics = call("/diagnostics")
    if diagnostics.get("failed_events", 0) > 0:
        raise RuntimeError(f"Replay completed with failed events: {diagnostics['failed_events']}")
    if replay.get("events_consumed", 0) < replay.get("events_published", 0):
        raise RuntimeError("Consumer lag remains after waiting.")
    return diagnostics


def generate_summary(**context) -> dict:
    diagnostics = context["ti"].xcom_pull(task_ids="validate_results", key="return_value")
    replay = diagnostics.get("replay", {})
    return {
        "status": diagnostics.get("replay_status"),
        "published": replay.get("events_published"),
        "consumed": replay.get("events_consumed"),
        "failed": diagnostics.get("failed_events"),
        "processing_rate_eps": diagnostics.get("processing_rate_eps"),
        "latest_batch": diagnostics.get("latest_batch"),
    }


def record_completion(**context) -> None:
    summary = context["ti"].xcom_pull(task_ids="generate_summary", key="return_value")
    print(json.dumps(summary, indent=2, default=str))


with DAG(
    dag_id="retail_replay_batch",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["retail", "kafka", "replay", "incremental"],
) as dag:
    determine = PythonOperator(task_id="determine_next_batch", python_callable=determine_next_batch)
    validate = PythonOperator(task_id="validate_batch", python_callable=validate_batch_selection)
    publish = PythonOperator(task_id="publish_batch", python_callable=publish_batch)
    wait_consumer = PythonOperator(task_id="wait_for_consumer", python_callable=wait_for_consumer_processing)
    refresh = PythonOperator(task_id="refresh_incremental_analytics", python_callable=refresh_incremental_analytics)
    validate_results_task = PythonOperator(task_id="validate_results", python_callable=validate_results)
    summary = PythonOperator(task_id="generate_summary", python_callable=generate_summary)
    completion = PythonOperator(task_id="record_completion", python_callable=record_completion)

    determine >> validate >> publish >> wait_consumer >> refresh >> validate_results_task >> summary >> completion

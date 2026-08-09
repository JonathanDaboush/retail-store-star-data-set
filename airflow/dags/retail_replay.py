"""Orchestrate one real retail replay batch through the public control plane."""
from datetime import datetime, timedelta
import json
import time
from urllib.request import Request, urlopen
from airflow import DAG
from airflow.operators.python import PythonOperator

API = "http://backend:8000"

def call(path, payload=None):
    data = json.dumps(payload).encode() if payload else None
    request = Request(f"{API}{path}", data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read())

def start_batch():
    # Reuse the API's replay state and Kafka publishing semantics rather than duplicating them in Airflow.
    return call("/replay/start", {"batch_size": 100, "interval_seconds": 5})

def verify_publication():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = call("/replay")
        if state["status"] == "failed":
            raise RuntimeError(state.get("last_error") or "Replay failed")
        if state.get("events_consumed", 0) >= state.get("events_published", 0) and state.get("events_published", 0):
            return state
        time.sleep(5)
    raise TimeoutError("Kafka consumer did not finish the published replay events within three minutes.")

with DAG(
    dag_id="retail_replay_batch",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["retail", "kafka", "replay"],
) as dag:
    trigger = PythonOperator(task_id="start_replay_batch", python_callable=start_batch)
    validate = PythonOperator(task_id="validate_replay_state", python_callable=verify_publication)
    trigger >> validate

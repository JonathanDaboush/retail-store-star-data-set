"""Business API and replay control plane for the retail star-schema demonstration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import text

from database import engine
from ml_functions.feature_engineering import create_all_ml_feature_tables, clean_all_ml_training_datasets
from ml_functions.final_start_schema_datasets import create_final_ml_datasets
from ml_functions.preprocess import preprocess_all_datasets

ROOT = Path(__file__).resolve().parent
EVENT_BANK = ROOT / "original_data" / "fact_sales_normalized.csv"
MODEL_DIR = ROOT / "models"
UPLOADS = ROOT / "data" / "uploads" / "original"
UPLOADS.mkdir(parents=True, exist_ok=True)

ML_MODEL_NAMES = ("forecast", "churn", "ltv", "demand", "recommendation")
MODEL_ARTIFACT_PATHS = {name: MODEL_DIR / f"{name}_model.pkl" for name in ML_MODEL_NAMES}
TARGET_COLUMNS = {
    "forecast": "future_revenue",
    "churn": "churn_label",
    "ltv": "customer_ltv",
    "demand": "total_units_sold",
}
ML_DATA_CACHE_TTL_SECONDS = 15
ML_FACT_COLUMNS = (
    "sales_sk",
    "sales_id",
    "customer_sk",
    "product_sk",
    "store_sk",
    "salesperson_sk",
    "campaign_sk",
    "sales_date",
    "total_amount",
)

app = FastAPI(title="Retail Operations API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

replay_lock = threading.Lock()
replay_thread: threading.Thread | None = None
log = logging.getLogger(__name__)


class ReplayRequest(BaseModel):
    batch_size: int = Field(default=100, ge=1, le=5000)
    interval_seconds: float = Field(default=5, ge=0.1, le=3600)
    batch_mode: Literal["events", "day", "week", "store"] = "events"
    batch_value: str | None = None

    @model_validator(mode="after")
    def validate_batch_selector(self) -> "ReplayRequest":
        if self.batch_mode == "events":
            return self
        if not self.batch_value:
            raise ValueError("batch_value is required when batch_mode is not 'events'.")
        if self.batch_mode == "day" and not re.match(r"^\d{4}-\d{2}-\d{2}$", self.batch_value):
            raise ValueError("batch_value must use YYYY-MM-DD for day mode.")
        if self.batch_mode == "week" and not re.match(r"^\d{4}-W\d{2}$", self.batch_value):
            raise ValueError("batch_value must use YYYY-Www for week mode.")
        if self.batch_mode == "store" and not str(self.batch_value).isdigit():
            raise ValueError("batch_value must be a numeric store_sk for store mode.")
        return self


class ReplayAction(BaseModel):
    action: Literal["pause", "resume", "stop"]


class MLPredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=500)


class MLRecommendRequest(BaseModel):
    customer_sk: int = Field(ge=1)
    top_n: int = Field(default=5, ge=1, le=20)


def setup() -> None:
    with engine.begin() as c:
        c.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS replay_state (
                    id TINYINT PRIMARY KEY,
                    status VARCHAR(16) NOT NULL,
                    next_offset INT NOT NULL DEFAULT 0,
                    batch_size INT NOT NULL DEFAULT 100,
                    interval_seconds DECIMAL(10,2) NOT NULL DEFAULT 5,
                    batch_mode VARCHAR(16) NOT NULL DEFAULT 'events',
                    batch_value VARCHAR(64) NULL,
                    total_events INT NOT NULL DEFAULT 0,
                    events_published INT NOT NULL DEFAULT 0,
                    events_consumed INT NOT NULL DEFAULT 0,
                    failed_events INT NOT NULL DEFAULT 0,
                    last_batch_size INT NOT NULL DEFAULT 0,
                    last_batch_published DATETIME NULL,
                    started_at DATETIME NULL,
                    last_error TEXT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
        c.execute(
            text(
                """
                INSERT IGNORE INTO replay_state
                (id,status,next_offset,batch_size,interval_seconds,batch_mode,batch_value,total_events,events_published,events_consumed,failed_events,last_batch_size,last_batch_published,started_at,updated_at)
                VALUES (1,'idle',0,100,5,'events',NULL,0,0,0,0,0,NULL,NULL,UTC_TIMESTAMP())
                """
            )
        )

        for migration in [
            "ALTER TABLE replay_state ADD COLUMN batch_mode VARCHAR(16) NOT NULL DEFAULT 'events'",
            "ALTER TABLE replay_state ADD COLUMN batch_value VARCHAR(64) NULL",
            "ALTER TABLE replay_state ADD COLUMN total_events INT NOT NULL DEFAULT 0",
            "ALTER TABLE replay_state ADD COLUMN events_consumed INT NOT NULL DEFAULT 0",
            "ALTER TABLE replay_state ADD COLUMN failed_events INT NOT NULL DEFAULT 0",
            "ALTER TABLE replay_state ADD COLUMN last_batch_size INT NOT NULL DEFAULT 0",
            "ALTER TABLE replay_state ADD COLUMN last_batch_published DATETIME NULL",
            "ALTER TABLE replay_state ADD COLUMN started_at DATETIME NULL",
        ]:
            try:
                c.execute(text(migration))
            except Exception:
                pass

        c.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS processed_events (
                    event_id VARCHAR(128) PRIMARY KEY,
                    transaction_id VARCHAR(128) NULL,
                    processed_at DATETIME NOT NULL
                )
                """
            )
        )
        c.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS failed_events (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    event_id VARCHAR(128) NULL,
                    dedupe_key VARCHAR(180) NULL UNIQUE,
                    reason VARCHAR(512) NOT NULL,
                    payload_json LONGTEXT NULL,
                    kafka_offset BIGINT NULL,
                    failed_at DATETIME NOT NULL
                )
                """
            )
        )
        for migration in [
            "ALTER TABLE failed_events ADD COLUMN dedupe_key VARCHAR(180) NULL",
            "ALTER TABLE failed_events ADD UNIQUE KEY ux_failed_events_dedupe (dedupe_key)",
        ]:
            try:
                c.execute(text(migration))
            except Exception:
                pass
        c.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS replay_batches (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    batch_mode VARCHAR(16) NOT NULL,
                    batch_value VARCHAR(64) NULL,
                    batch_size INT NOT NULL,
                    published_count INT NOT NULL,
                    started_offset INT NOT NULL,
                    ended_offset INT NOT NULL,
                    published_at DATETIME NOT NULL
                )
                """
            )
        )


setup()


@lru_cache(maxsize=1)
def event_bank_frame() -> pd.DataFrame:
    frame = pd.read_csv(EVENT_BANK)
    frame["sales_date"] = pd.to_datetime(frame["sales_date"], errors="coerce")
    frame = frame.dropna(subset=["sales_date"]).copy()
    frame["sales_day"] = frame["sales_date"].dt.strftime("%Y-%m-%d")
    frame["sales_week"] = frame["sales_date"].dt.strftime("%G-W%V")
    return frame


def scalar(sql: str, params: dict[str, Any] | None = None) -> Any:
    with engine.connect() as c:
        return c.execute(text(sql), params or {}).scalar()


def rows(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    with engine.connect() as c:
        return [dict(row._mapping) for row in c.execute(text(sql), params or {})]


_ml_model_cache: dict[str, dict[str, Any]] = {}
_ml_dataset_cache: dict[str, Any] = {"loaded_at": 0.0, "snapshot": None, "payload": None}
_ml_model_cache_lock = threading.Lock()
_ml_dataset_cache_lock = threading.Lock()


def fact_snapshot() -> dict[str, Any]:
    snapshot = rows(
        """
        SELECT COUNT(*) AS count_rows,
               MAX(sales_sk) AS max_sales_sk,
               MAX(sales_date) AS max_sales_date
        FROM fact_sales_normalized
        """
    )[0]
    return {
        "count_rows": int(snapshot.get("count_rows") or 0),
        "max_sales_sk": snapshot.get("max_sales_sk"),
        "max_sales_date": str(snapshot.get("max_sales_date") or ""),
    }


@lru_cache(maxsize=1)
def dimension_frames() -> dict[str, pd.DataFrame]:
    base = ROOT / "original_data"
    customers = pd.read_csv(base / "dim_customers.csv")
    products = pd.read_csv(base / "dim_products.csv")
    dates = pd.read_csv(base / "dim_dates.csv")
    if "full_date" in dates.columns:
        dates["full_date"] = pd.to_datetime(dates["full_date"], errors="coerce")
    return {"customers": customers, "products": products, "dates": dates}


def load_model_package(name: str) -> dict[str, Any]:
    path = MODEL_ARTIFACT_PATHS.get(name)
    if path is None:
        raise ValueError(f"Unsupported model: {name}")
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    with _ml_model_cache_lock:
        mtime = path.stat().st_mtime
        cached = _ml_model_cache.get(name)
        if cached and cached.get("mtime") == mtime:
            return cached["package"]
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        expected_digest = os.getenv(f"ML_MODEL_SHA256_{name.upper()}")
        if not expected_digest:
            raise ValueError(f"Missing digest configuration: ML_MODEL_SHA256_{name.upper()}")
        if digest.lower() != expected_digest.strip().lower():
            raise ValueError(f"Model digest check failed for {name}")
        package = pickle.loads(blob)
        if not isinstance(package, dict) or "model" not in package:
            raise ValueError(f"Invalid model package format for {name}")
        _ml_model_cache[name] = {"mtime": mtime, "package": package}
        return package


def ml_feature_frame(dataset_name: str, processed: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if dataset_name not in processed:
        raise ValueError(f"Dataset {dataset_name} is unavailable.")
    frame = processed[dataset_name].copy()
    target = TARGET_COLUMNS.get(dataset_name)
    if target:
        frame = frame.drop(columns=[target], errors="ignore")
    if frame.empty:
        raise ValueError(f"Dataset {dataset_name} has no rows after preprocessing.")
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.fillna(0.0)


def align_for_model(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    feature_names = expected_model_features(model)
    if feature_names is None:
        return frame
    aligned = frame.copy()
    for feature in feature_names:
        if feature not in aligned.columns:
            aligned[feature] = 0.0
    aligned = aligned[list(feature_names)]
    return aligned.fillna(0.0)


def expected_model_features(model: Any) -> list[str] | None:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return [str(feature) for feature in feature_names]

    fallback_names = getattr(model, "feature_name_", None)
    if fallback_names:
        return [str(feature) for feature in fallback_names]

    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            return [str(feature) for feature in booster.feature_name()]
        except Exception:  # noqa: BLE001
            return None
    return None


def load_ml_datasets() -> dict[str, Any]:
    now = time.time()
    snapshot = fact_snapshot()
    with _ml_dataset_cache_lock:
        cached_payload = _ml_dataset_cache.get("payload")
        if (
            cached_payload is not None
            and _ml_dataset_cache.get("snapshot") == snapshot
            and (now - float(_ml_dataset_cache.get("loaded_at") or 0.0)) < ML_DATA_CACHE_TTL_SECONDS
        ):
            return cached_payload

        facts = pd.read_sql(
            text(
                """
                SELECT sales_sk,sales_id,customer_sk,product_sk,store_sk,salesperson_sk,campaign_sk,sales_date,total_amount
                FROM fact_sales_normalized
                """
            ),
            con=engine,
        )
        if facts.empty:
            payload = {"snapshot": snapshot, "raw": {}, "processed": {}}
            _ml_dataset_cache.update({"loaded_at": now, "snapshot": snapshot, "payload": payload})
            return payload

        facts = facts[[col for col in ML_FACT_COLUMNS if col in facts.columns]].copy()
        facts["sales_date"] = pd.to_datetime(facts["sales_date"], errors="coerce")
        facts = facts.dropna(subset=["sales_date"])

        dims = dimension_frames()
        ml_features = create_all_ml_feature_tables(facts, dims["customers"], dims["products"])
        final_datasets = create_final_ml_datasets(ml_features, dims["customers"], dims["products"], dims["dates"])
        cleaned = clean_all_ml_training_datasets(final_datasets)
        processed, _encoders = preprocess_all_datasets(cleaned)

        payload = {"snapshot": snapshot, "raw": cleaned, "processed": processed}
        _ml_dataset_cache.update({"loaded_at": now, "snapshot": snapshot, "payload": payload})
        return payload


def model_feature_schema(name: str) -> list[str]:
    try:
        package = load_model_package(name)
        model = package.get("model")
        feature_names = expected_model_features(model)
        if feature_names is None:
            return []
        return feature_names
    except Exception:
        return []


def selected_frame(batch_mode: str, batch_value: str | None) -> pd.DataFrame:
    frame = event_bank_frame()
    if batch_mode == "events":
        return frame
    if batch_mode == "day":
        return frame[frame["sales_day"] == batch_value]
    if batch_mode == "week":
        return frame[frame["sales_week"] == batch_value]
    if batch_mode == "store":
        return frame[frame["store_sk"].astype(str) == str(batch_value)]
    raise ValueError(f"Unsupported batch_mode: {batch_mode}")


def normalized_replay_state() -> dict[str, Any]:
    state = rows(
        """
        SELECT status,next_offset,batch_size,interval_seconds,batch_mode,batch_value,total_events,
               events_published,events_consumed,failed_events,last_batch_size,last_batch_published,
               started_at,last_error,updated_at
        FROM replay_state WHERE id=1
        """
    )[0]
    total_events = int(state.get("total_events") or 0)
    next_offset = int(state.get("next_offset") or 0)
    state["events_remaining"] = max(0, total_events - next_offset)
    elapsed = None
    if state.get("started_at"):
        started = pd.to_datetime(state["started_at"], utc=True)
        elapsed = max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)
    state["processing_rate_eps"] = round((state.get("events_consumed") or 0) / elapsed, 3) if elapsed and elapsed >= 1.0 else 0.0
    state["lag_events"] = max(0, (state.get("events_published") or 0) - (state.get("events_consumed") or 0))
    return state


def start_worker_if_needed() -> None:
    global replay_thread
    with replay_lock:
        if replay_thread and replay_thread.is_alive():
            return
        replay_thread = threading.Thread(target=publish_batch_worker, daemon=True)
        replay_thread.start()


def build_replay_event(row: dict[str, Any], batch_mode: str, batch_value: str | None) -> dict[str, Any]:
    sales_date = row["sales_date"]
    if isinstance(sales_date, pd.Timestamp):
        sales_date = sales_date.isoformat()

    payload = {
        "sales_sk": int(row["sales_sk"]),
        "sales_id": str(row["sales_id"]),
        "customer_sk": int(row["customer_sk"]),
        "product_sk": int(row["product_sk"]),
        "store_sk": int(row["store_sk"]),
        "salesperson_sk": int(row["salesperson_sk"]),
        "campaign_sk": int(row["campaign_sk"]),
        "sales_date": sales_date,
        "total_amount": float(row["total_amount"]),
    }

    return {
        "event_id": f"sale-{payload['sales_sk']}",
        "transaction_id": payload["sales_id"],
        "event_type": "sale",
        "source": "original_data/fact_sales_normalized.csv",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "replay": {"batch_mode": batch_mode, "batch_value": batch_value},
        "payload": payload,
    }


def record_publish_failure(event: dict[str, Any], reason: str) -> None:
    dedupe_key = hashlib.sha256(f"{event.get('event_id')}|publish|{reason[:160]}".encode("utf-8")).hexdigest()
    with engine.begin() as c:
        inserted = c.execute(
            text(
                """
                INSERT IGNORE INTO failed_events (event_id, dedupe_key, reason, payload_json, kafka_offset, failed_at)
                VALUES (:event_id, :dedupe_key, :reason, :payload_json, NULL, UTC_TIMESTAMP())
                """
            ),
            {
                "event_id": event.get("event_id"),
                "dedupe_key": dedupe_key,
                "reason": reason[:512],
                "payload_json": json.dumps(event, default=str)[:65000],
            },
        ).rowcount
        if inserted:
            c.execute(
                text(
                    """
                    UPDATE replay_state
                    SET failed_events = failed_events + 1, updated_at = UTC_TIMESTAMP()
                    WHERE id=1
                    """
                )
            )


def publish_batch_worker() -> None:
    from controller.producer import send_event

    while True:
        try:
            state = normalized_replay_state()
            if state["status"] != "running":
                return

            batch_frame = selected_frame(state["batch_mode"], state["batch_value"])
            total_events = len(batch_frame)
            if state["next_offset"] >= total_events:
                with engine.begin() as c:
                    c.execute(
                        text(
                            """
                            UPDATE replay_state
                            SET status='completed', total_events=:total_events, updated_at=UTC_TIMESTAMP()
                            WHERE id=1
                            """
                        ),
                        {"total_events": total_events},
                    )
                return

            start_offset = state["next_offset"]
            batch_size = int(state["batch_size"])
            end_offset = min(start_offset + batch_size, total_events)
            chunk = batch_frame.iloc[start_offset:end_offset]

            published = 0
            for _, row in chunk.iterrows():
                event = build_replay_event(row.to_dict(), state["batch_mode"], state["batch_value"])
                sent = send_event(event)
                if not sent:
                    record_publish_failure(event, f"Kafka publish failed for event_id={event['event_id']}")
                    continue
                published += 1

            with engine.begin() as c:
                c.execute(
                    text(
                        """
                        UPDATE replay_state
                        SET next_offset=:next_offset,
                            total_events=:total_events,
                            events_published=events_published+:published,
                            last_batch_size=:published,
                            last_batch_published=UTC_TIMESTAMP(),
                            updated_at=UTC_TIMESTAMP(),
                            last_error=NULL
                        WHERE id=1
                        """
                    ),
                    {
                        "next_offset": end_offset,
                        "total_events": total_events,
                        "published": published,
                    },
                )
                c.execute(
                    text(
                        """
                        INSERT INTO replay_batches
                        (batch_mode, batch_value, batch_size, published_count, started_offset, ended_offset, published_at)
                        VALUES (:batch_mode, :batch_value, :batch_size, :published_count, :started_offset, :ended_offset, UTC_TIMESTAMP())
                        """
                    ),
                    {
                        "batch_mode": state["batch_mode"],
                        "batch_value": state["batch_value"],
                        "batch_size": batch_size,
                        "published_count": published,
                        "started_offset": start_offset,
                        "ended_offset": end_offset,
                    },
                )

            time.sleep(float(state["interval_seconds"]))
        except Exception as exc:  # noqa: BLE001
            with engine.begin() as c:
                c.execute(
                    text(
                        """
                        UPDATE replay_state
                        SET status='failed', last_error=:error, updated_at=UTC_TIMESTAMP()
                        WHERE id=1
                        """
                    ),
                    {"error": str(exc)[:2000]},
                )
            return


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        scalar("SELECT 1")
        return {"status": "Healthy", "database": "connected", "timestamp": datetime.now(timezone.utc)}
    except Exception as exc:  # noqa: BLE001
        log.exception("Health check failed")
        return {"status": "Failed", "database": "unavailable", "detail": "Database check failed"}


@app.get("/dashboard")
def dashboard() -> dict[str, Any]:
    totals = rows(
        """
        SELECT COUNT(*) orders,
               COALESCE(SUM(total_amount),0) revenue,
               COUNT(DISTINCT customer_sk) customers
        FROM fact_sales_normalized
        """
    )[0]
    trend = rows(
        """
        SELECT DATE(sales_date) date,
               ROUND(SUM(total_amount),2) revenue
        FROM fact_sales_normalized
        GROUP BY DATE(sales_date)
        ORDER BY date DESC
        LIMIT 30
        """
    )[::-1]
    products = rows(
        """
        SELECT p.product_name name,
               ROUND(SUM(f.total_amount),2) revenue
        FROM fact_sales_normalized f
        JOIN dim_products p ON p.product_sk=f.product_sk
        GROUP BY p.product_name
        ORDER BY revenue DESC
        LIMIT 5
        """
    )
    return {
        "kpis": totals,
        "revenue_trend": trend,
        "top_products": products,
        "freshness": scalar("SELECT MAX(sales_date) FROM fact_sales_normalized"),
    }


@app.get("/customers")
def customers() -> dict[str, Any]:
    return {
        "customers": rows(
            "SELECT customer_sk, customer_id, first_name, last_name, customer_segment FROM dim_customers LIMIT 100"
        )
    }


@app.get("/analytics")
def analytics(task: Literal["total_revenue", "revenue_by_category", "daily_revenue", "top_customers"] = Query(...)) -> dict[str, Any]:
    if task == "total_revenue":
        return {
            "task": task,
            "value": float(scalar("SELECT COALESCE(SUM(total_amount),0) FROM fact_sales_normalized") or 0),
            "unit": "USD",
        }
    if task == "revenue_by_category":
        data = rows(
            """
            SELECT p.category label,
                   ROUND(SUM(f.total_amount),2) value
            FROM fact_sales_normalized f
            JOIN dim_products p ON p.product_sk=f.product_sk
            GROUP BY p.category
            ORDER BY value DESC
            """
        )
    elif task == "daily_revenue":
        data = rows(
            """
            SELECT DATE(sales_date) label,
                   ROUND(SUM(total_amount),2) value
            FROM fact_sales_normalized
            GROUP BY DATE(sales_date)
            ORDER BY label
            """
        )
    else:
        data = rows(
            """
            SELECT CONCAT(c.first_name, ' ', c.last_name) label,
                   ROUND(SUM(f.total_amount),2) value
            FROM fact_sales_normalized f
            JOIN dim_customers c ON c.customer_sk=f.customer_sk
            GROUP BY c.customer_sk,c.first_name,c.last_name
            ORDER BY value DESC
            LIMIT 10
            """
        )
    return {"task": task, "rows": data, "empty": not data}


@app.get("/replay")
def replay_status() -> dict[str, Any]:
    return normalized_replay_state()


@app.get("/replay/options")
def replay_options() -> dict[str, Any]:
    frame = event_bank_frame()
    day_counts = frame.groupby("sales_day").size().sort_values(ascending=False).head(30)
    week_counts = frame.groupby("sales_week").size().sort_values(ascending=False).head(30)
    store_counts = frame.groupby("store_sk").size().sort_values(ascending=False).head(30)
    return {
        "events_total": int(len(frame)),
        "days": [{"value": day, "events": int(count)} for day, count in day_counts.items()],
        "weeks": [{"value": week, "events": int(count)} for week, count in week_counts.items()],
        "stores": [{"value": str(store), "events": int(count)} for store, count in store_counts.items()],
    }


@app.post("/replay/start")
def start_replay(request: ReplayRequest) -> dict[str, Any]:
    frame = selected_frame(request.batch_mode, request.batch_value)
    total_events = int(len(frame))
    if total_events == 0:
        raise HTTPException(422, "No events match the selected batch_mode and batch_value.")

    with engine.begin() as c:
        current = c.execute(text("SELECT status FROM replay_state WHERE id=1 FOR UPDATE")).scalar()
        if current == "running":
            raise HTTPException(409, "Replay is already running")

        c.execute(
            text(
                """
                UPDATE replay_state
                SET status='running',
                    next_offset=0,
                    batch_size=:batch_size,
                    interval_seconds=:interval_seconds,
                    batch_mode=:batch_mode,
                    batch_value=:batch_value,
                    total_events=:total_events,
                    events_published=0,
                    events_consumed=0,
                    failed_events=0,
                    last_batch_size=0,
                    last_batch_published=NULL,
                    started_at=UTC_TIMESTAMP(),
                    last_error=NULL,
                    updated_at=UTC_TIMESTAMP()
                WHERE id=1
                """
            ),
            {
                "batch_size": request.batch_size,
                "interval_seconds": request.interval_seconds,
                "batch_mode": request.batch_mode,
                "batch_value": request.batch_value,
                "total_events": total_events,
            },
        )

    start_worker_if_needed()

    return normalized_replay_state()


@app.post("/replay/control")
def control_replay(request: ReplayAction) -> dict[str, Any]:
    status = {"pause": "paused", "resume": "running", "stop": "stopped"}[request.action]
    with engine.begin() as c:
        c.execute(text("UPDATE replay_state SET status=:status,updated_at=UTC_TIMESTAMP() WHERE id=1"), {"status": status})

    if request.action == "resume":
        start_worker_if_needed()

    return normalized_replay_state()


@app.get("/diagnostics")
def diagnostics() -> dict[str, Any]:
    replay = normalized_replay_state()
    try:
        from controller.producer import kafka_available

        kafka = "Healthy" if kafka_available() else "Failed"
    except Exception:
        kafka = "Failed"

    airflow = "Failed"
    airflow_health_url = os.getenv("AIRFLOW_HEALTH_URL", "http://airflow-webserver:8080/health")
    try:
        with urlopen(airflow_health_url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        metadatabase_status = str(payload.get("metadatabase", {}).get("status", "")).lower()
        scheduler_status = str(payload.get("scheduler", {}).get("status", "")).lower()
        airflow = "Healthy" if metadatabase_status == "healthy" and scheduler_status == "healthy" else "Warning"
    except (OSError, URLError, ValueError):
        airflow = "Failed"

    latest_batch = rows(
        """
        SELECT batch_mode, batch_value, published_count, started_offset, ended_offset, published_at
        FROM replay_batches
        ORDER BY id DESC
        LIMIT 1
        """
    )

    replay_status_label = "Processing" if replay["status"] == "running" else replay["status"].capitalize()
    freshness = scalar("SELECT MAX(sales_date) FROM fact_sales_normalized")

    return {
        "api": "Healthy",
        "database": "Healthy",
        "kafka": kafka,
        "airflow": airflow,
        "replay_status": replay_status_label,
        "replay": replay,
        "processed_transactions": int(scalar("SELECT COUNT(*) FROM fact_sales_normalized") or 0),
        "failed_events": int(scalar("SELECT COUNT(*) FROM failed_events") or 0),
        "processing_rate_eps": replay["processing_rate_eps"],
        "data_freshness": freshness,
        "latest_batch": latest_batch[0] if latest_batch else None,
    }


@app.get("/ml/status")
def ml_status() -> dict[str, Any]:
    models = []
    for name in ML_MODEL_NAMES:
        path = MODEL_DIR / f"{name}_model.pkl"
        digest_configured = bool(os.getenv(f"ML_MODEL_SHA256_{name.upper()}"))
        models.append(
            {
                "model": name,
                "artifact_present": path.exists(),
                "artifact_path": str(path.relative_to(ROOT)) if path.exists() else None,
                "feature_count": len(model_feature_schema(name)),
                "digest_configured": digest_configured,
            }
        )
    endpoints_enabled = all(
        (MODEL_DIR / f"{name}_model.pkl").exists() and bool(os.getenv(f"ML_MODEL_SHA256_{name.upper()}"))
        for name in TARGET_COLUMNS
    )
    return {
        "models": models,
        "prediction_endpoints": endpoints_enabled,
        "note": "Prediction/reporting endpoints require model artifacts and matching ML_MODEL_SHA256_* environment variables.",
    }


@app.get("/ml/schema")
def ml_schema() -> dict[str, Any]:
    return {
        "models": [
            {
                "model": name,
                "features": model_feature_schema(name),
            }
            for name in ML_MODEL_NAMES
        ]
    }


@app.post("/ml/predict/{model_name}")
def ml_predict(model_name: str, request: MLPredictRequest) -> dict[str, Any]:
    if model_name not in ML_MODEL_NAMES:
        raise HTTPException(404, f"Unsupported model: {model_name}")
    try:
        package = load_model_package(model_name)
        model = package["model"]
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Failed loading model %s", model_name)
        raise HTTPException(500, f"Unable to load model {model_name}: {exc}") from exc

    frame = pd.DataFrame(request.records)
    if frame.empty:
        raise HTTPException(422, "records cannot be empty.")
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.fillna(0.0)
    aligned = align_for_model(model, frame)
    if aligned.empty:
        raise HTTPException(422, "No usable numeric features were provided.")

    try:
        if model_name == "recommendation":
            distances, indices = model.kneighbors(aligned, n_neighbors=min(10, len(aligned) + 1))
            return {
                "model": model_name,
                "rows": int(len(aligned)),
                "neighbors": [
                    {
                        "row_index": int(i),
                        "distances": [float(x) for x in distances[i]],
                        "indices": [int(x) for x in indices[i]],
                    }
                    for i in range(len(aligned))
                ],
            }

        predictions = model.predict(aligned)
        response: dict[str, Any] = {
            "model": model_name,
            "rows": int(len(predictions)),
            "predictions": [float(x) for x in np.asarray(predictions)],
        }
        if model_name == "churn" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(aligned)[:, 1]
            response["probabilities"] = [float(x) for x in np.asarray(probabilities)]
        return response
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed for model=%s", model_name)
        raise HTTPException(500, f"Inference failed for {model_name}: {exc}") from exc


@app.post("/ml/recommendations")
def ml_recommendations(request: MLRecommendRequest) -> dict[str, Any]:
    try:
        data = load_ml_datasets()
        raw = data["raw"].get("recommendation")
        processed = data["processed"].get("recommendation")
        if raw is None or processed is None or raw.empty or processed.empty:
            raise HTTPException(422, "Insufficient processed events for recommendations.")
        package = load_model_package("recommendation")
        model = package["model"]

        customer_mask = raw["customer_sk"].astype(int) == int(request.customer_sk)
        if not customer_mask.any():
            raise HTTPException(404, f"No interactions found for customer_sk={request.customer_sk}")

        customer_vectors = processed.loc[customer_mask].copy()
        if customer_vectors.empty:
            raise HTTPException(422, "Unable to build customer feature vector.")
        customer_vector = customer_vectors.mean(axis=0).to_frame().T
        aligned = align_for_model(model, customer_vector)

        neighbors = min(max(request.top_n * 4, request.top_n + 1), len(processed))
        distances, indices = model.kneighbors(aligned, n_neighbors=neighbors)
        index_list = [int(i) for i in indices[0]]
        distance_list = [float(d) for d in distances[0]]

        if len(processed) != len(raw):
            raise HTTPException(500, "Recommendation dataset alignment mismatch.")
        rec_rows = raw.reset_index(drop=True).iloc[index_list].copy()
        rec_rows["distance"] = distance_list
        rec_rows = rec_rows[rec_rows["customer_sk"].astype(int) != int(request.customer_sk)]
        if rec_rows.empty:
            return {"customer_sk": request.customer_sk, "recommendations": []}

        product_meta = dimension_frames()["products"][["product_sk", "product_name", "category", "brand"]].drop_duplicates("product_sk")
        ranked = (
            rec_rows.groupby("product_sk", as_index=False)
            .agg(score=("distance", "mean"), support=("distance", "size"))
            .sort_values(["score", "support"], ascending=[True, False])
            .head(request.top_n)
        )
        ranked = ranked.merge(product_meta, on="product_sk", how="left")
        recommendations = [
            {
                "product_sk": int(row["product_sk"]),
                "product_name": row.get("product_name"),
                "category": row.get("category"),
                "brand": row.get("brand"),
                "similarity_score": float(max(0.0, 1.0 - float(row["score"]))),
                "support": int(row["support"]),
            }
            for _, row in ranked.iterrows()
        ]
        return {
            "customer_sk": request.customer_sk,
            "recommendations": recommendations,
            "source_interactions": int(customer_mask.sum()),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Recommendation query failed for customer_sk=%s", request.customer_sk)
        raise HTTPException(500, f"Recommendation request failed: {exc}") from exc


@app.get("/ml/report")
def ml_report() -> dict[str, Any]:
    start = time.time()
    try:
        data = load_ml_datasets()
        if not data["processed"]:
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data_snapshot": data["snapshot"],
                "models": {},
                "note": "No processed fact data is available for model scoring yet.",
            }

        reports: dict[str, Any] = {}
        for name in TARGET_COLUMNS:
            try:
                package = load_model_package(name)
                model = package["model"]
                feature_frame = ml_feature_frame(name, data["processed"])
                aligned = align_for_model(model, feature_frame)
                if aligned.empty:
                    reports[name] = {"status": "unavailable", "reason": "No usable features"}
                    continue
                predictions = np.asarray(model.predict(aligned), dtype=float)
                base_report = {
                    "status": "ok",
                    "rows_scored": int(len(predictions)),
                    "mean_prediction": float(np.mean(predictions)) if len(predictions) else None,
                }

                if name == "forecast":
                    raw_forecast = data["raw"].get("forecast")
                    if raw_forecast is None or raw_forecast.empty:
                        reports[name] = {"status": "unavailable", "reason": "Forecast dataset is empty"}
                        continue
                    raw_forecast = raw_forecast.reset_index(drop=True)
                    if "date" in raw_forecast.columns:
                        parsed_dates = pd.to_datetime(raw_forecast["date"], errors="coerce")
                        latest_idx = int(parsed_dates.idxmax()) if parsed_dates.notna().any() else (len(raw_forecast) - 1)
                    else:
                        latest_idx = len(raw_forecast) - 1
                    if len(predictions):
                        latest_idx = min(max(0, latest_idx), len(predictions) - 1)
                    idx = int(np.argmax(predictions)) if len(predictions) else None
                    base_report.update(
                        {
                            "next_day_revenue_estimate": float(predictions[latest_idx]) if len(predictions) else None,
                            "peak_forecast": {
                                "date": str(raw_forecast.iloc[idx]["date"]) if idx is not None and idx < len(raw_forecast) else None,
                                "revenue": float(predictions[idx]) if idx is not None else None,
                            },
                        }
                    )
                elif name == "churn":
                    raw_churn = data["raw"].get("churn")
                    if raw_churn is None or raw_churn.empty:
                        reports[name] = {"status": "unavailable", "reason": "Churn dataset is empty"}
                        continue
                    raw_churn = raw_churn.reset_index(drop=True)
                    probs = np.asarray(model.predict_proba(aligned)[:, 1], dtype=float) if hasattr(model, "predict_proba") else predictions
                    ranked = pd.DataFrame(
                        {
                            "customer_sk": raw_churn.get("customer_sk"),
                            "first_name": raw_churn.get("first_name"),
                            "last_name": raw_churn.get("last_name"),
                            "risk": probs,
                        }
                    ).sort_values("risk", ascending=False).head(10)
                    base_report.update(
                        {
                            "high_risk_count": int((probs >= 0.7).sum()),
                            "avg_churn_probability": float(np.mean(probs)) if len(probs) else None,
                            "top_risk_customers": [
                                {
                                    "customer_sk": int(row["customer_sk"]) if pd.notna(row["customer_sk"]) else None,
                                    "name": f"{row.get('first_name') or ''} {row.get('last_name') or ''}".strip() or None,
                                    "risk": float(row["risk"]),
                                }
                                for _, row in ranked.iterrows()
                            ],
                        }
                    )
                elif name == "ltv":
                    raw_ltv = data["raw"].get("ltv")
                    if raw_ltv is None or raw_ltv.empty:
                        reports[name] = {"status": "unavailable", "reason": "LTV dataset is empty"}
                        continue
                    raw_ltv = raw_ltv.reset_index(drop=True)
                    ranked = pd.DataFrame(
                        {
                            "customer_sk": raw_ltv.get("customer_sk"),
                            "predicted_ltv": predictions,
                        }
                    ).sort_values("predicted_ltv", ascending=False).head(10)
                    base_report.update(
                        {
                            "top_customers": [
                                {
                                    "customer_sk": int(row["customer_sk"]) if pd.notna(row["customer_sk"]) else None,
                                    "predicted_ltv": float(row["predicted_ltv"]),
                                }
                                for _, row in ranked.iterrows()
                            ]
                        }
                    )
                elif name == "demand":
                    raw_demand = data["raw"].get("demand")
                    if raw_demand is None or raw_demand.empty:
                        reports[name] = {"status": "unavailable", "reason": "Demand dataset is empty"}
                        continue
                    raw_demand = raw_demand.reset_index(drop=True)
                    ranked = pd.DataFrame(
                        {
                            "product_sk": raw_demand.get("product_sk"),
                            "product_name": raw_demand.get("product_name"),
                            "predicted_units": predictions,
                        }
                    ).sort_values("predicted_units", ascending=False).head(10)
                    base_report.update(
                        {
                            "top_products": [
                                {
                                    "product_sk": int(row["product_sk"]) if pd.notna(row["product_sk"]) else None,
                                    "product_name": row.get("product_name"),
                                    "predicted_units": float(row["predicted_units"]),
                                }
                                for _, row in ranked.iterrows()
                            ]
                        }
                    )

                reports[name] = base_report
            except FileNotFoundError as exc:
                reports[name] = {"status": "missing_artifact", "reason": str(exc)}
            except Exception as exc:  # noqa: BLE001
                log.exception("Model report failed for %s", name)
                reports[name] = {"status": "failed", "reason": str(exc)}

        elapsed = round(time.time() - start, 3)
        log.info("Generated ML report in %ss", elapsed)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_snapshot": data["snapshot"],
            "elapsed_seconds": elapsed,
            "models": reports,
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("ML report generation failed")
        raise HTTPException(500, f"ML reporting failed: {exc}") from exc


@app.post("/uploads/preview")
async def preview_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(422, "Upload an Excel .xlsx or .xls file.")

    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "The file must be 25 MB or smaller.")

    digest = hashlib.sha256(content).hexdigest()
    original = UPLOADS / f"{digest}{suffix}"
    if not original.exists():
        original.write_bytes(content)

    try:
        frame = pd.read_excel(original)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"The Excel file could not be read: {exc}") from exc

    required_sales_columns = {
        "sales_id",
        "customer_sk",
        "product_sk",
        "store_sk",
        "salesperson_sk",
        "campaign_sk",
        "sales_date",
        "total_amount",
    }
    normalized = {str(c).strip().lower() for c in frame.columns}
    missing = sorted(required_sales_columns - normalized)
    duplicates = int(frame.duplicated().sum())

    issues = []
    if missing and len(normalized & required_sales_columns) >= 3:
        issues.append(f"Missing expected retail columns: {', '.join(missing)}")

    if "total_amount" in normalized:
        col = next((c for c in frame.columns if str(c).strip().lower() == "total_amount"), None)
        if col is not None:
            non_numeric = int(pd.to_numeric(frame[col], errors="coerce").isna().sum())
            if non_numeric:
                issues.append(f"{non_numeric} rows have non-numeric total_amount values.")

    if "sales_id" in normalized:
        col = next((c for c in frame.columns if str(c).strip().lower() == "sales_id"), None)
        if col is not None:
            duplicate_sales_ids = int(frame[col].astype(str).duplicated().sum())
            if duplicate_sales_ids:
                issues.append(f"{duplicate_sales_ids} duplicate sales_id values detected.")

    return {
        "dataset_id": digest,
        "filename": file.filename,
        "rows": len(frame),
        "columns": list(frame.columns),
        "types": {k: str(v) for k, v in frame.dtypes.items()},
        "sample": json.loads(frame.head(8).to_json(orient="records", date_format="iso")),
        "duplicate_rows": duplicates,
        "validation_errors": issues,
        "validation": "Preview completed. Original file is preserved unchanged.",
    }

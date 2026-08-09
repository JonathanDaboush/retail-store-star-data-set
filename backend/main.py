"""Business API and replay control plane for the retail star-schema demonstration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import text

from database import engine

ROOT = Path(__file__).resolve().parent
EVENT_BANK = ROOT / "original_data" / "fact_sales_normalized.csv"
UPLOADS = ROOT / "data" / "uploads" / "original"
UPLOADS.mkdir(parents=True, exist_ok=True)

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
        "airflow": "Configured",
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
    model_dir = ROOT / "models"
    models = []
    for name in ["forecast", "churn", "ltv", "demand", "recommendation"]:
        path = model_dir / f"{name}_model.pkl"
        models.append(
            {
                "model": name,
                "artifact_present": path.exists(),
                "artifact_path": str(path.relative_to(ROOT)) if path.exists() else None,
            }
        )
    return {
        "models": models,
        "prediction_endpoints": False,
        "note": "Model artifacts are present but this API currently exposes model availability only.",
    }


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

"""Kafka worker for idempotent retail transaction ingestion."""

import json
import logging
import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict

from kafka import KafkaConsumer
from sqlalchemy import text

from database import engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TOPIC = os.getenv("KAFKA_TOPIC", "retail-events-kafka")
REQUIRED_PAYLOAD_FIELDS = {
    "sales_sk",
    "sales_id",
    "customer_sk",
    "product_sk",
    "store_sk",
    "salesperson_sk",
    "campaign_sk",
    "sales_date",
    "total_amount",
}


def setup_tables() -> None:
    with engine.begin() as conn:
        conn.execute(
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
        conn.execute(
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
        conn.execute(
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
        conn.execute(
            text(
                """
                INSERT IGNORE INTO replay_state
                (id,status,next_offset,batch_size,interval_seconds,batch_mode,batch_value,total_events,events_published,events_consumed,failed_events,last_batch_size,last_batch_published,started_at,last_error,updated_at)
                VALUES (1,'idle',0,100,5,'events',NULL,0,0,0,0,0,NULL,NULL,NULL,UTC_TIMESTAMP())
                """
            )
        )


def validate_event(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("Event must be a JSON object")
    if event.get("event_type") != "sale":
        raise ValueError("Unsupported event type")
    if "event_id" not in event:
        raise ValueError("event_id is required")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    missing = sorted(REQUIRED_PAYLOAD_FIELDS - set(payload.keys()))
    if missing:
        raise ValueError(f"Missing payload fields: {', '.join(missing)}")
    return payload


def record_failure(message_value: Any, reason: str, kafka_offset: int | None) -> None:
    event_id = message_value.get("event_id") if isinstance(message_value, dict) else None
    payload_json = json.dumps(message_value, default=str)[:65000]
    fingerprint_source = f"{event_id}|{kafka_offset}|{reason[:160]}"
    dedupe_key = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    with engine.begin() as conn:
        inserted = conn.execute(
            text(
                """
                INSERT IGNORE INTO failed_events (event_id, dedupe_key, reason, payload_json, kafka_offset, failed_at)
                VALUES (:event_id, :dedupe_key, :reason, :payload_json, :kafka_offset, UTC_TIMESTAMP())
                """
            ),
            {
                "event_id": event_id,
                "dedupe_key": dedupe_key,
                "reason": reason[:512],
                "payload_json": payload_json,
                "kafka_offset": kafka_offset,
            },
        ).rowcount
        if inserted:
            conn.execute(
                text(
                    """
                    UPDATE replay_state
                    SET failed_events = failed_events + 1, updated_at = UTC_TIMESTAMP()
                    WHERE id = 1
                    """
                )
            )


def process(event: Dict[str, Any]) -> str:
    payload = validate_event(event)
    with engine.begin() as conn:
        inserted = conn.execute(
            text(
                """
                INSERT IGNORE INTO processed_events (event_id, transaction_id, processed_at)
                VALUES (:event_id, :transaction_id, UTC_TIMESTAMP())
                """
            ),
            {
                "event_id": str(event["event_id"]),
                "transaction_id": str(event.get("transaction_id") or payload.get("sales_id") or ""),
            },
        ).rowcount

        if inserted == 0:
            return "duplicate"

        conn.execute(
            text(
                """
                INSERT INTO fact_sales_normalized
                (sales_sk, sales_id, customer_sk, product_sk, store_sk, salesperson_sk, campaign_sk, sales_date, total_amount)
                VALUES (:sales_sk,:sales_id,:customer_sk,:product_sk,:store_sk,:salesperson_sk,:campaign_sk,:sales_date,:total_amount)
                ON DUPLICATE KEY UPDATE sales_sk = VALUES(sales_sk)
                """
            ),
            payload,
        )
        conn.execute(
            text(
                """
                UPDATE replay_state
                SET events_consumed = events_consumed + 1, updated_at = UTC_TIMESTAMP()
                WHERE id = 1
                """
            )
        )
    return "processed"


def run() -> None:
    setup_tables()
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
        group_id="retail-fact-loader",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    log.info("Consumer started topic=%s at %s", TOPIC, datetime.now(timezone.utc).isoformat())
    while True:
        for _topic_partition, messages in consumer.poll(timeout_ms=1000, max_records=100).items():
            for message in messages:
                try:
                    result = process(message.value)
                    consumer.commit()
                    if result == "duplicate":
                        log.info("Skipped duplicate event at offset=%s", message.offset)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Event processing failed at offset=%s", message.offset)
                    try:
                        record_failure(message.value, str(exc), message.offset)
                        consumer.commit()
                    except Exception:  # noqa: BLE001
                        log.exception("Failed to persist malformed event at offset=%s", message.offset)


if __name__ == "__main__":
    run()

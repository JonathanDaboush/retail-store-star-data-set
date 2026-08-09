import json
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient
from kafka.errors import KafkaError

load_dotenv()

log = logging.getLogger(__name__)
producer = None


def get_producer() -> KafkaProducer:
    global producer
    if producer is None:
        producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            key_serializer=lambda value: value.encode("utf-8") if value else None,
            retries=5,
            acks="all",
            linger_ms=50,
            request_timeout_ms=15000,
        )
    return producer


def kafka_available() -> bool:
    admin = KafkaAdminClient(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
        request_timeout_ms=2000,
    )
    try:
        admin.list_topics()
        return True
    finally:
        admin.close()


def send_event(event: Dict[str, Any], timeout_seconds: int = 15) -> bool:
    topic = os.getenv("KAFKA_TOPIC", "retail-events-kafka")
    key = str(event.get("transaction_id") or event.get("event_id") or "")
    try:
        future = get_producer().send(topic, event, key=key)
        metadata = future.get(timeout=timeout_seconds)
        log.info(
            "Published event_id=%s topic=%s partition=%s offset=%s",
            event.get("event_id"),
            metadata.topic,
            metadata.partition,
            metadata.offset,
        )
        return True
    except KafkaError:
        log.exception("Kafka publish failed for event_id=%s", event.get("event_id"))
        return False


def send_row_event(table_name: str, row: Dict[str, Any]) -> bool:
    payload = {"event_type": "row_loaded", "table": table_name, "payload": row}
    return send_event(payload)

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient
import json
import os
from dotenv import load_dotenv

load_dotenv()

producer = None

def get_producer():
    global producer
    if producer is None:
        producer = KafkaProducer(
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            key_serializer=lambda value: value.encode("utf-8") if value else None,
        )
    return producer

def kafka_available():
    """Probe broker metadata; diagnostics must not infer Kafka health from UI state."""
    admin = KafkaAdminClient(bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"), request_timeout_ms=2000)
    try:
        admin.list_topics()
        return True
    finally:
        admin.close()

def send_event(event):
    get_producer().send(
        os.getenv("KAFKA_TOPIC", "retail-events-kafka"),
        event, key=str(event.get("event_id", ""))
    )
    get_producer().flush()

def send_row_event(table_name, row):
    payload = {
        "event_type": "row_loaded",
        "table": table_name,
        "payload": row
    }
    send_event(payload)

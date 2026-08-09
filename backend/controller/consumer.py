"""Kafka worker for idempotent retail transaction ingestion."""
import json, os, logging
from kafka import KafkaConsumer
from sqlalchemy import text
from database import engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
topic = os.getenv("KAFKA_TOPIC", "retail-events-kafka")

def process(event):
    if event.get("event_type") != "sale":
        raise ValueError("Unsupported event type")
    sale = event["payload"]
    with engine.begin() as conn:
        # The primary key makes Kafka's at-least-once delivery safe for business totals.
        conn.execute(text("""
            INSERT INTO fact_sales_normalized
            (sales_sk, sales_id, customer_sk, product_sk, store_sk, salesperson_sk, campaign_sk, sales_date, total_amount)
            VALUES (:sales_sk,:sales_id,:customer_sk,:product_sk,:store_sk,:salesperson_sk,:campaign_sk,:sales_date,:total_amount)
            ON DUPLICATE KEY UPDATE sales_sk = VALUES(sales_sk)
        """), sale)
        inserted = conn.execute(text("INSERT IGNORE INTO processed_events (event_id, processed_at) VALUES (:event_id, UTC_TIMESTAMP())"), {"event_id": event["event_id"]}).rowcount
        if inserted:
            conn.execute(text("UPDATE replay_state SET events_consumed=events_consumed+1, updated_at=UTC_TIMESTAMP() WHERE id=1"))

def run():
    consumer = KafkaConsumer(topic, bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
        group_id="retail-fact-loader", enable_auto_commit=False, auto_offset_reset="earliest",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")))
    for message in consumer:
        try:
            process(message.value)
            consumer.commit()  # Only acknowledge after the database transaction commits.
        except Exception:
            log.exception("Event processing failed at offset %s", message.offset)

if __name__ == "__main__":
    run()

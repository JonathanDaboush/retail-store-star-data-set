from kafka import KafkaProducer
import json
import os
from dotenv import load_dotenv

load_dotenv()

producer = KafkaProducer(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
    value_serializer=lambda x: json.dumps(x).encode("utf-8")
)

def send_event(event):
    producer.send(
        os.getenv("KAFKA_TOPIC", "retail-events-kafka"),
        event
    )
    producer.flush()

def send_row_event(table_name, row):
    payload = {
        "event_type": "row_loaded",
        "table": table_name,
        "payload": row
    }
    send_event(payload)
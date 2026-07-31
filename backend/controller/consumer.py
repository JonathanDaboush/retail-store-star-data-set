from kafka import KafkaConsumer
import json
import os
from dotenv import load_dotenv

load_dotenv()

consumer = KafkaConsumer(
    os.getenv("KAFKA_TOPIC", "retail-events-kafka"),
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("Kafka consumer started...")

for message in consumer:
    event = message.value
    print("Processing event:", event)
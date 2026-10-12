"""
Запуск:
    docker compose up -d
    uvicorn outbox_inbox:api --port 8000
    python outbox_inbox.py relay
    python outbox_inbox.py worker

    curl -F "file=@bus.jpg" http://localhost:8000/predict
    curl http://localhost:8000/tasks/<task_id>
"""

import argparse
import json
import logging
import sqlite3
import uuid
from pathlib import Path

import pika
from fastapi import FastAPI, HTTPException, UploadFile

from recognition import load_model, predict

DB_PATH = "outbox_inbox.db"
RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
QUEUE = "images"
UPLOADS = Path("uploads")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-6s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("pika").setLevel(logging.WARNING)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks  (id TEXT PRIMARY KEY, image_path TEXT, status TEXT, result TEXT);
CREATE TABLE IF NOT EXISTS outbox (message_id TEXT PRIMARY KEY, payload TEXT, published INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS inbox  (message_id TEXT PRIMARY KEY);
"""


def connect_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)

    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)

    return db


def connect_rabbitmq():
    channel = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL)).channel()
    channel.queue_declare(QUEUE, durable=True)

    return channel


db = connect_db()
UPLOADS.mkdir(exist_ok=True)

api = FastAPI()
api_log = logging.getLogger("api")


@api.post("/predict")
async def create_task(file: UploadFile):
    task_id = str(uuid.uuid4())

    image_path = UPLOADS / f"{task_id}.jpg"
    image_path.write_bytes(await file.read())

    message_id = str(uuid.uuid4())
    message = json.dumps({"task_id": task_id, "image_path": str(image_path)})

    with db:
        db.execute(
            "INSERT INTO tasks (id, image_path, status) VALUES (?, ?, 'PENDING')",
            (task_id, str(image_path)),
        )
        db.execute(
            "INSERT INTO outbox (message_id, payload) VALUES (?, ?)",
            (message_id, message),
        )

    api_log.info("task %s: задача и сообщение %s сохранены в одной транзакции", task_id[:8], message_id[:8])

    return {"task_id": task_id}


@api.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = db.execute("SELECT status, result FROM tasks WHERE id = ?", (task_id,)).fetchone()
    
    if task is None:
        raise HTTPException(404, "task not found")
    
    return {"status": task["status"], "result": json.loads(task["result"] or "null")}

def run_relay() -> None:
    log = logging.getLogger("relay")

    channel = connect_rabbitmq()
    channel.confirm_delivery()

    log.info("подключен к RabbitMQ, слежу за outbox")

    while True:
        messages = db.execute(
            "SELECT message_id, payload FROM outbox WHERE published = 0 ORDER BY rowid"
        ).fetchall()

        if messages:
            log.info("в outbox %d неотправленных сообщений", len(messages))

        for message in messages:
            channel.basic_publish(
                exchange="",
                routing_key=QUEUE,
                body=message["payload"],
                properties=pika.BasicProperties(
                    message_id=message["message_id"],
                    delivery_mode=pika.DeliveryMode.Persistent,
                ),
            )

            with db:
                db.execute("UPDATE outbox SET published = 1 WHERE message_id = ?", (message["message_id"],))

            log.info("сообщение %s отправлено, брокер подтвердил, отмечено в outbox", message["message_id"][:8])

        channel.connection.sleep(0.5)

def already_processed(message_id: str) -> bool:
    return db.execute("SELECT 1 FROM inbox WHERE message_id = ?", (message_id,)).fetchone() is not None


def save_result(message_id: str, task_id: str, status: str, result: object) -> None:
    with db:
        db.execute("INSERT INTO inbox (message_id) VALUES (?)", (message_id,))
        db.execute(
            "UPDATE tasks SET status = ?, result = ? WHERE id = ?",
            (status, json.dumps(result), task_id),
        )


def run_worker() -> None:
    log = logging.getLogger("worker")
    model = load_model()

    channel = connect_rabbitmq()
    channel.basic_qos(prefetch_count=1)

    log.info("жду сообщения из очереди %s", QUEUE)

    def handle(ch, method, properties, body):
        message_id = properties.message_id
        task = json.loads(body)
        log.info("получено сообщение %s (task %s)", message_id[:8], task["task_id"][:8])

        # Дубликат тоже надо ack-нуть, иначе при prefetch_count=1 воркер встанет навсегда
        if already_processed(message_id):
            log.warning("сообщение %s уже есть в inbox — дубликат, пропускаю", message_id[:8])
            ch.basic_ack(method.delivery_tag)
            return

        try:
            status, result = "SUCCESS", predict(model, task["image_path"])
            log.info("распознано объектов: %d", len(result["detections"]))
        except Exception as error:  # битая картинка не должна крутиться в очереди вечно
            log.exception("не удалось обработать task %s", task["task_id"][:8])
            status, result = "FAILURE", {"error": str(error)}

        try:
            save_result(message_id, task["task_id"], status, result)
            log.info("результат и запись в inbox сохранены в одной транзакции")
        except sqlite3.IntegrityError:
            log.warning("сообщение %s параллельно обработал другой воркер — пропускаю", message_id[:8])

        ch.basic_ack(method.delivery_tag)
        log.info("ack %s", message_id[:8])

    channel.basic_consume(QUEUE, handle)
    channel.start_consuming()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["relay", "worker"])
    args = parser.parse_args()

    if args.role == "relay":
        run_relay()
    else:
        run_worker()

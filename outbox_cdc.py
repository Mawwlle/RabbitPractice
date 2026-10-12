"""
Запуск:
    docker compose up -d --build
    uvicorn outbox_cdc:api --port 8000
    python outbox_cdc.py relay
    python outbox_cdc.py worker

    curl -F "file=@bus.jpg" http://localhost:8000/predict
    curl http://localhost:8000/tasks/<task_id>
"""

import argparse
import json
import logging
import select
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pika
import psycopg2
import psycopg2.extras
import psycopg2.pool
from fastapi import FastAPI, HTTPException, UploadFile

from recognition import load_model, predict

PG_DSN = "postgresql://app:app@localhost:5432/app"
RABBITMQ_URL = "amqp://guest:guest@localhost:5672/"
QUEUE = "images"
SLOT = "outbox_relay"
UPLOADS = Path("uploads")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-6s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("pika").setLevel(logging.WARNING)

psycopg2.extras.register_uuid()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id         uuid PRIMARY KEY,
    image_path text NOT NULL,
    status     text NOT NULL,
    result     jsonb
);

-- published нет: «что уже отправлено» знает слот репликации, а не таблица
CREATE TABLE IF NOT EXISTS outbox (
    id             uuid PRIMARY KEY,
    aggregate_type text        NOT NULL,
    aggregate_id   text        NOT NULL,  -- по нему партиционируют, чтобы сохранить порядок событий одной сущности
    type           text        NOT NULL,
    payload        jsonb       NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS inbox (
    message_id   uuid PRIMARY KEY,
    processed_at timestamptz NOT NULL DEFAULT now()
);
"""


def init_db() -> None:
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute(SCHEMA)
        # Слот создаём вместе со схемой, а не при старте relay: всё, что записано в WAL
        # после создания слота, Postgres сохранит до тех пор, пока relay это не прочитает.
        # Если бы слот создавал relay, вставки, сделанные до его первого запуска, потерялись бы.
        cur.execute(
            """
            SELECT pg_create_logical_replication_slot(%s, 'wal2json')
            WHERE NOT EXISTS (SELECT 1 FROM pg_replication_slots WHERE slot_name = %s)
            """,
            (SLOT, SLOT),
        )

    conn.close()


def connect_rabbitmq():
    channel = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL)).channel()
    channel.queue_declare(QUEUE, durable=True)

    return channel


init_db()
UPLOADS.mkdir(exist_ok=True)

pool = psycopg2.pool.ThreadedConnectionPool(1, 10, PG_DSN)


@contextmanager
def transaction():
    conn = pool.getconn()

    try:
        with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:  # commit или rollback
            yield cur
    finally:
        pool.putconn(conn)


api = FastAPI(title="YOLO + outbox (CDC)")
api_log = logging.getLogger("api")


#psycopg2 синхронный (можете попробовать сделать это через asyncpg)
@api.post("/predict")
def create_task(file: UploadFile):
    task_id = uuid.uuid4()

    image_path = UPLOADS / f"{task_id}.jpg"
    image_path.write_bytes(file.file.read())

    message_id = uuid.uuid4()
    payload = {"task_id": str(task_id), "image_path": str(image_path)}

    with transaction() as cur:
        cur.execute(
            "INSERT INTO tasks (id, image_path, status) VALUES (%s, %s, 'PENDING')",
            (task_id, str(image_path)),
        )
        cur.execute(
            """
            INSERT INTO outbox (id, aggregate_type, aggregate_id, type, payload)
            VALUES (%s, 'task', %s, 'TaskCreated', %s)
            """,
            (message_id, str(task_id), json.dumps(payload)),
        )

    api_log.info("task %s: задача и событие %s закоммичены одной транзакцией", str(task_id)[:8], str(message_id)[:8])

    return {"task_id": task_id}


@api.get("/tasks/{task_id}")
def get_task(task_id: uuid.UUID):
    with transaction() as cur:
        cur.execute("SELECT status, result FROM tasks WHERE id = %s", (task_id,))
        task = cur.fetchone()

    if task is None:
        raise HTTPException(404, "task not found")

    return task


def run_relay() -> None:
    log = logging.getLogger("relay")

    channel = connect_rabbitmq()
    channel.confirm_delivery()

    repl = psycopg2.connect(PG_DSN, connection_factory=psycopg2.extras.LogicalReplicationConnection)
    stream = repl.cursor()
    stream.start_replication(
        slot_name=SLOT,
        decode=True,
        options={
            "format-version": "2",          # одно JSON-сообщение на каждую изменённую строку
            "include-transaction": "false",  # без отдельных сообщений BEGIN/COMMIT
            "add-tables": "public.outbox",   # остальные таблицы нас не интересуют
            "actions": "insert",             # и только вставки
        },
    )

    log.info("читаю WAL через слот %s", SLOT)

    def publish(msg) -> None:
        change = json.loads(msg.payload)
        row = {column["name"]: column["value"] for column in change["columns"]}

        channel.basic_publish(
            exchange="",
            routing_key=QUEUE,
            body=row["payload"].encode(),  # jsonb wal2json отдаёт строкой с JSON
            properties=pika.BasicProperties(
                message_id=row["id"],
                type=row["type"],
                headers={"aggregate_id": row["aggregate_id"]},
                delivery_mode=pika.DeliveryMode.Persistent,
            ),
        )

        # Брокер подтвердил приём (confirm_delivery) — только теперь двигаем позицию слота.
        # Упадём между publish и send_feedback — после рестарта сообщение придёт ещё раз, его отсечёт inbox.
        msg.cursor.send_feedback(flush_lsn=msg.data_start)

        log.info("LSN %s: событие %s отправлено, брокер подтвердил", msg.data_start, row["id"][:8])

    cleanup_at = 0.0

    while True:
        msg = stream.read_message()  # неблокирующий; keepalive для Postgres шлёт сам

        if msg is not None:
            publish(msg)
            continue

        channel.connection.process_data_events(time_limit=0)  # heartbeat-ы RabbitMQ, пока ждём WAL

        # Строки outbox после коммита уже не нужны relay: событие лежит в WAL и удерживается слотом.
        # Таблицу оставляем как журнал для отладки и чистим с первого дня, как советует статья.
        if time.monotonic() > cleanup_at:
            with transaction() as cur:
                cur.execute("DELETE FROM outbox WHERE created_at < now() - interval '1 hour'")
            cleanup_at = time.monotonic() + 60

        select.select([stream], [], [], 1.0)  # спим, пока Postgres не пришлёт что-нибудь


def run_worker() -> None:
    log = logging.getLogger("worker")
    model = load_model()

    channel = connect_rabbitmq()
    channel.basic_qos(prefetch_count=1)

    log.info("жду сообщения из очереди %s", QUEUE)

    def already_processed(message_id: str) -> bool:
        with transaction() as cur:
            cur.execute("SELECT 1 FROM inbox WHERE message_id = %s", (message_id,))
            return cur.fetchone() is not None

    def save_result(message_id: str, task_id: str, status: str, result: object) -> bool:
        with transaction() as cur:
            cur.execute("INSERT INTO inbox (message_id) VALUES (%s) ON CONFLICT DO NOTHING", (message_id,))
            if cur.rowcount == 0:  # параллельный воркер успел первым
                return False

            cur.execute(
                "UPDATE tasks SET status = %s, result = %s WHERE id = %s",
                (status, json.dumps(result), task_id),
            )
            return True

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

        if save_result(message_id, task["task_id"], status, result):
            log.info("результат и запись в inbox сохранены в одной транзакции")
        else:
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

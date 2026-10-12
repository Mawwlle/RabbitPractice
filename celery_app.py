"""
    docker compose up -d
    celery -A celery_app worker --pool=solo --loglevel=info
    uvicorn celery_app:api --port 8000

    curl -F "file=@bus.jpg" http://localhost:8000/predict     # -> {"task_id": "..."}
    curl http://localhost:8000/tasks/<task_id>                # -> статус и результат
"""

import uuid
from pathlib import Path

from celery import Celery
from celery.result import AsyncResult
from fastapi import FastAPI, UploadFile

from recognition import load_model, predict

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

celery = Celery(
    "yolo",
    broker="amqp://guest:guest@localhost:5672//",  # задачи лежат в RabbitMQ
    backend="redis://localhost:6379/0",            # статусы и результаты — в Redis
)
celery.conf.task_track_started = True

model = None  # загружаем в воркере при первой задаче, а не при импорте (API модель не нужна)


@celery.task
def predict_task(image_path: str):
    global model # НУ ПРОСТИТЕ МЕНЯ НУ Я ТОЖЕ ЧЕЛОВЕК

    if model is None:
        model = load_model()

    return predict(model, image_path)


api = FastAPI(title="YOLO + Celery")


@api.post("/predict")
async def create_task(file: UploadFile):
    # картинку кладём на диск, а в очередь отправляем только путь к ней
    path = UPLOAD_DIR / f"{uuid.uuid4()}{Path(file.filename or '').suffix or '.jpg'}"
    path.write_bytes(await file.read())

    task = predict_task.delay(str(path))
    return {"task_id": task.id}


@api.get("/tasks/{task_id}")
async def get_task(task_id: str):
    result = AsyncResult(task_id, app=celery)

    return {
        "task_id": task_id,
        "status": result.status,  # PENDING → STARTED → SUCCESS / FAILURE
        "result": result.result if result.successful() else None,
        "error": str(result.result) if result.failed() else None,
    }


@api.get("/health")
async def health():
    return {"status": "ok"}

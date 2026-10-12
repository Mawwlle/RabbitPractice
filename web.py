import io

from fastapi import FastAPI, UploadFile
from PIL import Image

from recognition import load_model, predict

app = FastAPI(title="YOLO без очереди")
model = load_model()


@app.post("/predict")
async def predict_image(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    return predict(model, image)


@app.get("/health")
async def health():
    return {"status": "ok"}

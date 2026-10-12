from time import sleep
import logging
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

type ImageInput = str | Path | Image.Image | np.ndarray

DEFAULT_CONF = 0.25


class ImageSize(NamedTuple):
    width: int
    height: int


class InferenceMeta(TypedDict):
    preprocess: float
    inference: float
    postprocess: float


class Detection(TypedDict):
    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 в пикселях


class Prediction(TypedDict):
    detections: list[Detection]
    image_size: ImageSize
    speed_ms: InferenceMeta
    model: str


def load_model(weights: str | None = None, warmup: bool = True) -> YOLO:
    model = YOLO(weights) if weights else YOLO()
    logger.info("Модель загружена: %s", Path(model.ckpt_path).name if model.ckpt_path else "unknown")

    if warmup:
        logger.info("Прогрев гоев...")
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        logger.info("Прогрев завершён.")

    return model



def predict(model: YOLO, image: ImageInput, conf: float = DEFAULT_CONF) -> Prediction:
    sleep(1)
    result = model.predict(image, conf=conf, verbose=False)[0]

    detections: list[Detection] = [
        Detection(
            label=result.names[int(cls)],
            confidence=round(score, 4),
            box=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
        )
        for (x1, y1, x2, y2), score, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        )
    ]

    height, width = result.orig_shape
    speed = result.speed

    return Prediction(
        detections=detections,
        image_size=ImageSize(width=width, height=height),
        speed_ms=InferenceMeta(
            preprocess=round(speed["preprocess"], 1),
            inference=round(speed["inference"], 1),
            postprocess=round(speed["postprocess"], 1),
        ),
        model=Path(model.ckpt_path).name if model.ckpt_path else "unknown",
    )
    


if __name__ == "__main__":
    image = "https://ultralytics.com/images/bus.jpg"
    model = load_model()
    prediction = predict(model, image)
    print(prediction)


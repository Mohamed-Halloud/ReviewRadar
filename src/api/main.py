import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager

from src.model.inference import load_model, predict_batch

from .schemas import PredictionRequest, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["tokenizer"], models["model"] = load_model()
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "ok"}


labels = {0: "negative", 1: "neutral", 2: "positive"}


@app.post("/predict", response_model=PredictionResponse)
def prediction(request: PredictionRequest):
    review = request.review

    start = time.time()
    try:
        labels_pred, confidences = predict_batch(
            [review], models["tokenizer"], models["model"]
        )
        label = labels_pred[0]
        confidence = confidences[0]
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="prediction failed") from e
    latency = time.time() - start

    logger.info(
        f"Prediction made: input={review[:50]!r}, class={labels[label]}, "
        f"confidence={confidence:.2f}, latency={latency:.3f}s"
    )

    return PredictionResponse(review=review, label=label, review_class=labels[label])
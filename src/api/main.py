import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from model.inference import tokenizer, model, predict_batch
from api.schemas import PredictionRequest, PredictionResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

models = {}

# Load the model once at startup, clear it on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    models["model"] = model
    
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "ok"}

labels = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

@app.post("/predict", response_model=PredictionResponse)
def prediction(request: PredictionRequest):
    review = request.review

    if review is None:
        logger.warning("Prediction request rejected: no review provided")
        raise HTTPException(status_code=404, detail="review non trouvé")

    start = time.time()
    try:
        # Run inference on a single-item batch
        labels_pred, confidences = predict_batch([review], tokenizer, models["model"])
        label = labels_pred[0]
        confidence = confidences[0]
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="prediction failed")
    latency = time.time() - start

    # Log the request outcome: input, predicted class, confidence, and time taken
    logger.info(f"Prediction made: input={review[:50]!r}, class={labels[label]}, confidence={confidence:.2f}, latency={latency:.3f}s")

    return PredictionResponse(
        review=review,
        label=label,
        review_class=labels[label]
    )
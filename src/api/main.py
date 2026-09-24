from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from model.inference import tokenizer, model, predict_batch
from api.schemas import PredictionRequest, PredictionResponse

models = {}

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
        raise HTTPException(status_code=404, detail="review non trouvé")

    label = predict_batch([review], tokenizer, models["model"])[0]

    return PredictionResponse(
        review=review,
        label=label,
        review_class=labels[label]
    )
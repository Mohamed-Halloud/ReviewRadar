from pydantic import BaseModel

class PredictionRequest(BaseModel):
    review: str

class PredictionResponse(BaseModel):
    review: str
    label: int
    review_class: str
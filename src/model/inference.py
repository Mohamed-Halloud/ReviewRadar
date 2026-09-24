import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "model", "best_model")

MAX_LENGTH = 128
BATCH_SIZE = 16
TORCH_THREADS = 1


def load_model():
    torch.set_num_threads(TORCH_THREADS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return tokenizer, model


tokenizer, model = load_model()


def predict_batch(text, tokenizer, model):
    with torch.no_grad():
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).tolist()
    return prediction
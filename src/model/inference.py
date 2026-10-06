import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Resolve the model path relative to this file, so it works regardless of cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "model", "best_model")

MAX_LENGTH = 128
BATCH_SIZE = 16
TORCH_THREADS = 1


def load_model():
    # Limit CPU threads to keep inference predictable/lightweight
    torch.set_num_threads(TORCH_THREADS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    # Quantize linear layers to int8 for faster, lighter CPU inference
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return tokenizer, model


# Load once at import time so the model isn't reloaded on every prediction
tokenizer, model = load_model()


def predict_batch(text, tokenizer, model):
    with torch.no_grad():
        # Tokenize and pad/truncate the whole batch at once
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        logits = model(**inputs).logits
        # Convert logits to class probabilities
        probs = torch.softmax(logits, dim=1)
        # For each item, take the highest probability (confidence) and its class index (prediction)
        confidence, prediction = probs.max(dim=1)
    return prediction.tolist(), confidence.tolist()
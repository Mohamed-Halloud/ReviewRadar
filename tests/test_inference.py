import math

import torch

from src.model.inference import MAX_LENGTH, predict_batch


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:

    def __init__(self, logits):
        self._logits = torch.tensor(logits)

    def __call__(self, **inputs):
        return FakeOutput(self._logits)


def make_tokenizer(calls=None):
    def fake_tokenizer(text, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return {"input_ids": torch.zeros(len(text), 1, dtype=torch.long)}

    return fake_tokenizer


def test_one_result_per_text():
    model = FakeModel([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    preds, confs = predict_batch(["a", "b"], make_tokenizer(), model)
    assert len(preds) == 2
    assert len(confs) == 2


def test_prediction_is_argmax():
    model = FakeModel([[0.1, 0.2, 5.0], [5.0, 0.2, 0.1]])
    preds, _ = predict_batch(["a", "b"], make_tokenizer(), model)
    assert preds == [2, 0]


def test_confidence_is_softmax_probability():
    logits = [[0.0, 0.0, 0.0]]
    _, confs = predict_batch(["a"], make_tokenizer(), FakeModel(logits))
    assert math.isclose(confs[0], 1 / 3, rel_tol=1e-5)


def test_confidence_in_valid_range():
    model = FakeModel([[3.0, -2.0, 0.5], [-1.0, 4.0, 0.0]])
    _, confs = predict_batch(["a", "b"], make_tokenizer(), model)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_tokenizer_called_with_truncation():
    calls = []
    model = FakeModel([[1.0, 0.0, 0.0]])
    predict_batch(["a"], make_tokenizer(calls), model)
    assert calls[0]["truncation"] is True
    assert calls[0]["max_length"] == MAX_LENGTH
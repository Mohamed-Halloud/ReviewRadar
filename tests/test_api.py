import pytest
from fastapi.testclient import TestClient

from src.api import main


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setattr(main, "load_model", lambda: ("fake_tokenizer", "fake_model"))
    with TestClient(main.app) as c:
        yield c
    mp.undo()


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("idx,name", [(0, "negative"), (1, "neutral"), (2, "positive")])
def test_label_mapping(client, monkeypatch, idx, name):
    monkeypatch.setattr(main, "predict_batch", lambda texts, tok, mod: ([idx], [0.9]))
    response = client.post("/predict", json={"review": "anything"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == idx
    assert body["review_class"] == name


@pytest.mark.parametrize("payload", [{}, {"review": 123}, {"review": None}])
def test_invalid_payload(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_model_failure_returns_500(client, monkeypatch):
    def boom(*args):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(main, "predict_batch", boom)
    assert client.post("/predict", json={"review": "x"}).status_code == 500
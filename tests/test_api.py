import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_negative_review(client):
    response = client.post("/predict", json={"review": "this is a very bad book!"})

    assert response.status_code == 200
    body = response.json()
    assert body["review_class"] == "negative"


def test_positive_review(client):
    response = client.post("/predict", json={"review": "I loved this book, absolutely amazing!"})

    assert response.status_code == 200
    body = response.json()
    assert body["review_class"] == "positive"


def test_missing_review_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422  # FastAPI/Pydantic validation error


def test_wrong_type(client):
    response = client.post("/predict", json={"review": 123})
    assert response.status_code == 422
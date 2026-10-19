import os
import sys

import joblib
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from typing import get_args  # noqa: E402

from api.main import CATEGORICAL_COLUMNS, MODEL_PATH, CreditApplication, app  # noqa: E402

VALID_APPLICATION = {
    "age": 35,
    "sex": "male",
    "job": 2,
    "housing": "own",
    "saving_accounts": "little",
    "checking_account": "moderate",
    "credit_amount": 3000,
    "duration": 24,
    "purpose": "car",
}


@pytest.fixture(scope="module")
def client():
    # Le "with" declenche le lifespan, donc le chargement du modele
    with TestClient(app) as c:
        yield c


def test_model_file_contents():
    artifacts = joblib.load(MODEL_PATH)
    assert set(artifacts) == {"model", "feature_columns", "threshold"}
    assert list(artifacts["model"].feature_names_in_) == artifacts["feature_columns"]
    assert 0 < artifacts["threshold"] < 1


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_application(client):
    response = client.post("/predict", json=VALID_APPLICATION)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["prediction"] in ("good", "default")
    assert body["decision"] == ("reject" if body["prediction"] == "default" else "approve")
    assert (body["probability_of_default"] >= body["threshold"]) == (body["prediction"] == "default")


@pytest.mark.parametrize(
    "field, value",
    [
        (field, value)
        for field in CATEGORICAL_COLUMNS.values()
        for value in get_args(CreditApplication.model_fields[field].annotation)
    ],
)
def test_predict_every_category(client, field, value):
    # Chaque categorie autorisee (y compris celles de reference) doit etre scoree sans erreur 500
    response = client.post("/predict", json={**VALID_APPLICATION, field: value})
    assert response.status_code == 200, response.json()


@pytest.mark.parametrize(
    "field, value",
    [
        ("age", 15),
        ("sex", "other"),
        ("job", 7),
        ("credit_amount", -100),
        ("duration", 0),
        ("purpose", "casino"),
    ],
)
def test_predict_invalid_application(client, field, value):
    response = client.post("/predict", json={**VALID_APPLICATION, field: value})
    assert response.status_code == 422


def test_predict_missing_field(client):
    application = dict(VALID_APPLICATION)
    del application["age"]
    response = client.post("/predict", json=application)
    assert response.status_code == 422


def test_predict_batch(client):
    other = {**VALID_APPLICATION, "sex": "female", "housing": "rent", "purpose": "business"}
    response = client.post("/predict/batch", json=[VALID_APPLICATION, other])
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    # Un client seul et le meme client dans un lot doivent avoir le meme score
    single = client.post("/predict", json=VALID_APPLICATION).json()
    assert body[0] == single


def test_predict_batch_empty(client):
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 422

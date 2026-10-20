import pytest
from fastapi.testclient import TestClient

from api.main import app

PROFILE = {
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
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_known_profile_stays_stable(client):
    body = client.post("/predict", json=PROFILE).json()
    # Valeur figee apres le premier lancement : verrouille le chemin
    # entrainement -> service. Si ce test tombe, les deux ont diverge.
    assert body["probability_of_default"] == pytest.approx(0.3622, abs=1e-4)

def test_unknown_purpose_is_rejected(client):
    assert client.post("/predict", json={**PROFILE, "purpose": "crypto"}).status_code == 422


def test_age_below_18_is_rejected(client):
    assert client.post("/predict", json={**PROFILE, "age": 15}).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json=[]).status_code == 422
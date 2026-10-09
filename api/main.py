import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.joblib")

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Modele introuvable -> {MODEL_PATH}. Lancez d'abord : python src/train_rf.py"
        )
    artifacts.update(joblib.load(MODEL_PATH))
    if "threshold" not in artifacts:
        raise RuntimeError("Modele obsolete (sans seuil calibre). Relancez : python src/train_rf.py")
    yield
    artifacts.clear()


app = FastAPI(
    title="Credit Scoring API",
    description="Prediction de la probabilite de defaut (German Credit Data, Random Forest calibre).",
    version="1.0.0",
    lifespan=lifespan,
)


class CreditApplication(BaseModel):
    """Donnees brutes d'un client, au meme format que data/processed/german_credit_clean_updated.csv."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        },
    )

    age: int = Field(..., ge=18, le=100)
    sex: Literal["male", "female"]
    job: Literal[0, 1, 2, 3] = Field(..., description="Niveau de qualification (0 a 3)")
    housing: Literal["own", "rent", "free"]
    saving_accounts: Literal["none", "little", "moderate", "quite rich", "rich"]
    checking_account: Literal["none", "little", "moderate", "rich"]
    credit_amount: float = Field(..., gt=0)
    duration: int = Field(..., gt=0, description="Duree du pret en mois")
    purpose: Literal[
        "business",
        "car",
        "domestic appliances",
        "education",
        "furniture/equipment",
        "radio/TV",
        "repairs",
        "vacation/others",
    ]


class ScoringResult(BaseModel):
    probability_of_default: float
    prediction: Literal["good", "default"]
    decision: Literal["approve", "reject"]
    threshold: float


def build_features(applications: list[CreditApplication]) -> pd.DataFrame:
    """Reproduit features.py : ratio mensualite + one-hot, aligne sur les colonnes d'entrainement."""
    df = pd.DataFrame(
        [
            {
                "Age": a.age,
                "Sex": a.sex,
                "Job": a.job,
                "Housing": a.housing,
                "Saving accounts": a.saving_accounts,
                "Checking account": a.checking_account,
                "Credit amount": a.credit_amount,
                "Duration": a.duration,
                "Purpose": a.purpose,
            }
            for a in applications
        ]
    )
    df["Installment_Per_Month"] = df["Credit amount"] / df["Duration"]
    # Pas de drop_first ici : le reindex supprime les categories de reference
    # et ajoute (a 0) celles absentes du lot.
    df_encoded = pd.get_dummies(df)
    return df_encoded.reindex(columns=artifacts["feature_columns"], fill_value=0)


def score(applications: list[CreditApplication]) -> list[ScoringResult]:
    X = build_features(applications)
    probabilities = artifacts["model"].predict_proba(X)[:, 1]
    threshold = artifacts["threshold"]
    results = []
    for p in probabilities:
        is_default = p >= threshold
        results.append(
            ScoringResult(
                probability_of_default=round(float(p), 4),
                prediction="default" if is_default else "good",
                decision="reject" if is_default else "approve",
                threshold=round(threshold, 4),
            )
        )
    return results


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in artifacts}


@app.post("/predict", response_model=ScoringResult)
def predict(application: CreditApplication):
    return score([application])[0]


@app.post("/predict/batch", response_model=list[ScoringResult])
def predict_batch(applications: list[CreditApplication]):
    if not applications:
        raise HTTPException(status_code=422, detail="La liste de clients est vide.")
    return score(applications)

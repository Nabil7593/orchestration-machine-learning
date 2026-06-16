"""API d'inference d'un modele de classification (FastAPI).

Seance 12 - TP FastAPI
    Lancement : depuis credit_score/
        PYTHONPATH=src uvicorn api:app --reload
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    logger.info("Chargement du modele depuis %s", model_path)
    ml["model"] = joblib.load(model_path)
    logger.info("Modele charge avec succes")
    yield
    ml.clear()
    logger.info("Modele dechargé")


app = FastAPI(title="Credit Score Classification API", version="0.1.0", lifespan=lifespan)


class Features(BaseModel):
    # Numeriques
    Age: float = Field(..., ge=0, description="Age du client")
    Annual_Income: float = Field(..., ge=0, description="Revenu annuel")
    Monthly_Inhand_Salary: float = Field(..., ge=0, description="Salaire mensuel net")
    Num_Bank_Accounts: float = Field(..., ge=0, description="Nombre de comptes bancaires")
    Num_Credit_Card: float = Field(..., ge=0, description="Nombre de cartes de credit")
    Interest_Rate: float = Field(..., ge=0, description="Taux d'interet (%)")
    Num_of_Loan: float = Field(..., ge=0, description="Nombre de prets en cours")
    Delay_from_due_date: float = Field(..., ge=0, description="Retard moyen de paiement (jours)")
    Num_of_Delayed_Payment: float = Field(..., ge=0, description="Nombre de paiements en retard")
    Changed_Credit_Limit: float = Field(..., description="Variation de la limite de credit (%)")
    Num_Credit_Inquiries: float = Field(..., ge=0, description="Nombre de demandes de credit")
    Outstanding_Debt: float = Field(..., ge=0, description="Dette en cours")
    Credit_Utilization_Ratio: float = Field(
        ..., ge=0, le=100, description="Taux d'utilisation du credit (%)"
    )
    Total_EMI_per_month: float = Field(..., ge=0, description="Total des mensualites EMI")
    Amount_invested_monthly: float = Field(..., ge=0, description="Montant investi par mois")
    Monthly_Balance: float = Field(..., description="Solde mensuel")
    # Categorielles
    Occupation: str = Field(..., description="Profession du client")
    Credit_Mix: str = Field(..., description="Mix de credit (Good/Standard/Bad)")
    Payment_of_Min_Amount: str = Field(..., description="Paiement du montant minimum (Yes/No)")
    Payment_Behaviour: str = Field(..., description="Comportement de paiement")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 35.0,
                    "Annual_Income": 50000.0,
                    "Monthly_Inhand_Salary": 4000.0,
                    "Num_Bank_Accounts": 3.0,
                    "Num_Credit_Card": 4.0,
                    "Interest_Rate": 15.0,
                    "Num_of_Loan": 2.0,
                    "Delay_from_due_date": 5.0,
                    "Num_of_Delayed_Payment": 3.0,
                    "Changed_Credit_Limit": 2.5,
                    "Num_Credit_Inquiries": 2.0,
                    "Outstanding_Debt": 1500.0,
                    "Credit_Utilization_Ratio": 25.0,
                    "Total_EMI_per_month": 150.0,
                    "Amount_invested_monthly": 200.0,
                    "Monthly_Balance": 500.0,
                    "Occupation": "Engineer",
                    "Credit_Mix": "Standard",
                    "Payment_of_Min_Amount": "No",
                    "Payment_Behaviour": "High_spent_Small_value_payments",
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}

"""Tests de l'API FastAPI (modele mocke — pas besoin du .joblib)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api

VALID_PAYLOAD = {
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


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    with patch("joblib.load", return_value=mock_model):
        with TestClient(api.app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_prediction_and_probability(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_input_returns_422(client):
    r = client.post("/predict", json={"Age": -999})
    assert r.status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert "version" in r.json()

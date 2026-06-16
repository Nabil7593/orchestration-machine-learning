"""Script de prediction standalone (sans API).

Charge le modele depuis models/model.joblib et effectue une prediction
sur un exemple de client.

Usage :
    python prediction.py
    python prediction.py --proba   # affiche aussi la probabilite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"

EXAMPLE = {
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


def predict(data: dict, show_proba: bool = False) -> None:
    if not MODEL_PATH.exists():
        print(f"Erreur : modele introuvable ({MODEL_PATH})", file=sys.stderr)
        print("Lancez d'abord : make train-models", file=sys.stderr)
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([data])
    proba = float(model.predict_proba(row)[0, 1])
    label = "Good (1)" if proba >= 0.5 else "Standard/Poor (0)"

    print(f"Prediction  : {label}")
    if show_proba:
        print(f"Probabilite : {proba:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction standalone credit score")
    parser.add_argument("--proba", action="store_true", help="Affiche la probabilite")
    args = parser.parse_args()
    predict(EXAMPLE, show_proba=args.proba)


if __name__ == "__main__":
    main()

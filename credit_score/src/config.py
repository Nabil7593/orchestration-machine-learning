"""Configuration centrale du projet de classification.

C'est le SEUL fichier a adapter pour brancher votre propre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs
colonnes via ces constantes. Voir tp/TP_S0_projet_personnel.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]   # credit_score/
load_dotenv(ROOT / ".env")

# TODO (S0-1) : chemin vers votre fichier de donnees (CSV) place dans data/
DATA_PATH = ROOT / "src" / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"

# TODO (S0-2) : nom de la colonne cible binaire (valeurs 0/1)
TARGET = "Credit_Score"   # Good=1, Standard/Poor=0

# TODO (S0-3) : colonnes numeriques de votre dataset
NUMERIC_FEATURES: list[str] = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

# TODO (S0-4) : colonnes categorielles (peut rester vide : [])
CATEGORICAL_FEATURES: list[str] = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "credit-score-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-score-classifier")

MLFLOW_EXPERIMENT_DESCRIPTION = (
    "Classification binaire du score de credit (Good=1 / Standard-Poor=0). "
    "Dataset Kaggle - 100 000 clients, 20 features, desequilibre 18% Good. "
    "Comparaison Random Forest / XGBoost / LightGBM avec GridSearchCV. "
    "Metrique principale : roc_auc."
)

MLFLOW_EXPERIMENT_TAGS: dict[str, str] = {
    "project": "credit-score-classification",
    "team": "mlops-esgi",
    "dataset": "kaggle-credit-score",
    "task": "binary-classification",
    "models": "random_forest,xgboost,lightgbm",
}

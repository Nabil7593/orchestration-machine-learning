"""Pre-processing pipeline complet pour le dataset Credit Score Classification.

Toutes les transformations sont ici :
  - Nettoyage des valeurs parasites ("_", "NA"...)
  - Binarisation de la cible  : Good=1, Standard/Poor=0
  - Numerique  (16 col.) : nettoyage -> imputation mediane  -> StandardScaler
  - Categoriel  (4 col.) : imputation mode -> OneHotEncoder
  - Reste (ID, Name, SSN...) : supprime (remainder='drop')
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

# Valeurs parasites presentes dans le dataset Kaggle brut
_DIRTY = ["_", "NA", "#F%$D@*&8", ""]


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit Credit_Score en binaire : Good=1, Standard/Poor=0."""
    df = df.copy()
    df[TARGET] = (df[TARGET] == "Good").astype(int)
    return df


def _clean_numeric(X: np.ndarray) -> np.ndarray:
    """Remplace les valeurs parasites par NaN et convertit en float."""
    df = pd.DataFrame(X).replace(_DIRTY, float("nan"))
    return df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def build_preprocessor() -> ColumnTransformer:
    # Numerique : nettoyage -> imputation mediane -> normalisation
    numeric_pipeline = Pipeline([
        ("clean",   FunctionTransformer(_clean_numeric)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Categoriel : imputation mode -> encodage one-hot
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",   # supprime ID, Name, SSN, Type_of_Loan, Credit_History_Age...
    )

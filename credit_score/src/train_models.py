"""Comparaison de modeles de classification - Credit Score Classification.

Seance 7 - TP AutoML & SHAP
    Compare trois familles de modeles (Random Forest, XGBoost, LightGBM),
    chacune optimisee par GridSearchCV, et persiste le meilleur dans models/.
    Chaque modele est suivi dans MLflow (un run par modele, sous un run parent
    "compare-models") et le meilleur est enregistre dans le Model Registry.

    Dataset desequilibre (18% Good) : class_weight/scale_pos_weight actives
    sur chaque modele. Metrique principale : roc_auc.

Lancement :
    PYTHONPATH=src python -m train_models
    PYTHONPATH=src python -m train_models --cv 3 --scoring roc_auc
    PYTHONPATH=src python -m train_models --no-mlflow
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import MODEL_DIR, MODEL_NAME, RANDOM_STATE
from data import load_data, split
from evaluation import log_shap_summary
from features import binarize_target, build_preprocessor
from tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Le ColumnTransformer renvoie un tableau numpy sans noms de colonnes lors du
# scoring interne de la validation croisee : on neutralise cet avertissement.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# Ratio classes pour compenser le desequilibre (82 172 / 17 828 ≈ 4.6)
_SCALE_POS_WEIGHT = 4.6


@dataclass
class ModelSpec:
    name: str
    estimator: ClassifierMixin
    param_grid: dict


def build_model_specs() -> list[ModelSpec]:
    """Trois modeles adaptes au credit scoring desequilibre."""
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=_SCALE_POS_WEIGHT,
                n_jobs=-1,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                verbose=-1,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


@dataclass
class FitResult:
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    logger.info("Optimisation de %s (cv=%d, scoring=%s)", spec.name, cv, scoring)

    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    result = FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )
    logger.info(
        "%s -> cv_%s=%.4f  f1=%.4f  roc_auc=%.4f",
        spec.name,
        scoring,
        result.cv_score,
        result.f1,
        result.roc_auc,
    )
    return result


def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)

        # S7-4a : hyperparametres et metriques
        mlflow.log_params({"cv": cv, "scoring": scoring, **result.best_params})
        mlflow.log_metrics(
            {
                f"cv_{scoring}": result.cv_score,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )

        # Matrice de confusion
        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Standard/Poor", "Good"]).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Rapport de classification
        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(
            str, classification_report(y_test, result.preds, target_names=["Standard/Poor", "Good"])
        )
        mlflow.log_text(report_text, "classification_report.txt")
        logger.info("\n%s", report_text)

        # S7-4b : SHAP summary plot
        log_shap_summary(result.best_estimator, x_test, result.name)

        # Enregistrement du modele
        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )


def train_all(cv: int = 5, scoring: str = "roc_auc", use_mlflow: bool = True) -> list[FitResult]:
    df = load_data()
    df = binarize_target(df)
    x_train, x_test, y_train, y_test = split(df)

    logger.info("Train : %d lignes | Test : %d lignes", len(x_train), len(x_test))
    logger.info("Classes - 0: %d | 1 (Good): %d", (y_test == 0).sum(), (y_test == 1).sum())

    experiment_id = None
    if use_mlflow:
        experiment_id = setup_experiment()

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleur modele : %s (roc_auc=%.4f  f1=%.4f)", best.name, best.roc_auc, best.f1)

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models", experiment_id=experiment_id):
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            log_dataset(df, context="training")
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=register_as)
        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde : %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparaison RF / XGBoost / LightGBM - Credit Score"
    )
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis CV")
    parser.add_argument("--scoring", type=str, default="roc_auc", help="Metrique GridSearchCV")
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive MLflow (sans serveur de tracking)",
    )
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()

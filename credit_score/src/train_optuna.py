"""Optimisation d'hyperparametres avec Optuna.

Seance 6 - TP Optuna
    Ce module optimise les hyperparametres de trois familles de modeles
    (Random Forest, XGBoost, LightGBM) avec Optuna (sampler TPE), compare
    leurs performances et persiste le meilleur dans `models/model.joblib`.

Chaque famille est suivie dans MLflow (un run par famille, imbrique sous un
run parent) et la meilleure est enregistree dans le Model Registry.

Lancement :
    PYTHONPATH=src python -m train_optuna
    PYTHONPATH=src python -m train_optuna --n-trials 20 --cv 3
    PYTHONPATH=src python -m train_optuna --no-mlflow
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import optuna.samplers
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import MODEL_DIR, MODEL_NAME, RANDOM_STATE
from data import load_data, split
from evaluation import log_shap_summary
from features import binarize_target, build_preprocessor
from tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SCALE_POS_WEIGHT = 4.6


@dataclass
class ModelSpec:
    name: str
    suggest_params: Callable
    build_estimator: Callable[[dict], ClassifierMixin]


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="random_forest",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            },
            build_estimator=lambda params: RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
                **params,
            ),
        ),
        ModelSpec(
            name="xgboost",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            },
            build_estimator=lambda params: XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=_SCALE_POS_WEIGHT,
                n_jobs=-1,
                **params,
            ),
        ),
        ModelSpec(
            name="lightgbm",
            suggest_params=lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
            },
            build_estimator=lambda params: cast(
                ClassifierMixin,
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    verbose=-1,
                    **params,
                ),
            ),
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def objective(trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    params = spec.suggest_params(trial)
    pipeline = build_pipeline(spec.build_estimator(params))
    scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, spec, x_train, y_train, cv),
        n_trials=n_trials,
    )
    return study


@dataclass
class FamilyResult:
    spec: ModelSpec
    study: Any
    best_pipeline: Pipeline
    test_roc_auc: float
    preds: np.ndarray


def optimize_family(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
) -> FamilyResult:
    logger.info("Optimisation de %s (n_trials=%d, cv=%d)", spec.name, n_trials, cv)
    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)

    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_roc_auc = float(roc_auc_score(y_test, proba))

    logger.info(
        "%s : cv_roc_auc=%.3f  test_roc_auc=%.3f  params=%s",
        spec.name,
        study.best_value,
        test_roc_auc,
        study.best_params,
    )
    return FamilyResult(
        spec=spec,
        study=study,
        best_pipeline=best_pipeline,
        test_roc_auc=test_roc_auc,
        preds=preds,
    )


def describe_registered_version(
    name: str,
    version: int,
    result: FamilyResult,
    n_trials: int,
    cv: int,
) -> None:
    client = mlflow.MlflowClient()
    description = (
        f"Modele {result.spec.name} optimise par Optuna "
        f"(sampler TPE, n_trials={n_trials}, cv={cv}).\n"
        f"Meilleurs hyperparametres : {result.study.best_params}\n"
        f"Metriques : cv_roc_auc={result.study.best_value:.3f}, "
        f"test_roc_auc={result.test_roc_auc:.3f}"
    )
    client.update_model_version(name=name, version=str(version), description=description)
    tags = {
        "model_family": result.spec.name,
        "search_method": "optuna-tpe",
        "n_trials": str(n_trials),
        "cv": str(cv),
        "cv_roc_auc": f"{result.study.best_value:.4f}",
        "test_roc_auc": f"{result.test_roc_auc:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name=name, version=str(version), key=key, value=value)


def log_family_to_mlflow(
    result: FamilyResult,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
    register_as: str | None = None,
) -> None:
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.set_tag("model_family", result.spec.name)
        mlflow.set_tag("sampler", "TPE")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)

        for trial in result.study.trials:
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("cv_roc_auc", trial.value)

        mlflow.log_params(result.study.best_params)
        mlflow.log_metric("cv_roc_auc", result.study.best_value)
        mlflow.log_metric("test_roc_auc", result.test_roc_auc)

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Standard/Poor", "Good"]).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.spec.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(
            str, classification_report(y_test, result.preds, target_names=["Standard/Poor", "Good"])
        )
        mlflow.log_text(report_text, "classification_report.txt")
        logger.info("\n%s", report_text)

        log_shap_summary(result.best_pipeline, x_test, result.spec.name)

        signature = infer_signature(x_test, result.best_pipeline.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_pipeline,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        if register_as and model_info.registered_model_version is not None:
            describe_registered_version(
                name=register_as,
                version=int(model_info.registered_model_version),
                result=result,
                n_trials=n_trials,
                cv=cv,
            )


def optimize(n_trials: int = 30, cv: int = 5, use_mlflow: bool = True) -> list[FamilyResult]:
    df = load_data()
    df = binarize_target(df)
    x_train, x_test, y_train, y_test = split(df)

    logger.info("Train : %d lignes | Test : %d lignes", len(x_train), len(x_test))
    logger.info("Classes - 0: %d | 1 (Good): %d", (y_test == 0).sum(), (y_test == 1).sum())

    experiment_id = None
    if use_mlflow:
        experiment_id = setup_experiment()

    results = [
        optimize_family(spec, x_train, y_train, x_test, y_test, n_trials=n_trials, cv=cv)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleure famille : %s (test_roc_auc=%.3f)", best.spec.name, best.test_roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare", experiment_id=experiment_id):
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
            mlflow.set_tag("best_model", best.spec.name)
            log_dataset(df, context="training")
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_family_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=register_as)
        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimisation Optuna RF / XGBoost / LightGBM - Credit Score"
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Nombre d'essais Optuna")
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisee")
    parser.add_argument(
        "--no-mlflow", dest="use_mlflow", action="store_false", help="Desactive le suivi MLflow"
    )
    args = parser.parse_args()
    optimize(n_trials=args.n_trials, cv=args.cv, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()

"""Entrainement du modele baseline - Credit Score Classification.

Seance 5 - TP MLflow Tracking
    Ce script entraine et evalue un modele SANS aucun suivi d'experience.
    Votre mission : instrumenter cet entrainement avec MLflow (voir les TODO).
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from config import MODEL_DIR
from data import load_data, split
from features import binarize_target, build_preprocessor

# TODO (S5-1) : importer mlflow et mlflow.sklearn


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("clf", LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced")),
    ])


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    # --- Chargement et preparation ---
    df = load_data()
    df = binarize_target(df)
    x_train, x_test, y_train, y_test = split(df)

    print(f"\n{'='*50}")
    print(f"  Credit Score Classifier - Baseline")
    print(f"{'='*50}")
    print(f"Train : {x_train.shape[0]} lignes  |  Test : {x_test.shape[0]} lignes")
    print(f"Classes - 0 (Standard/Poor) : {(y_test == 0).sum()}  |  1 (Good) : {(y_test == 1).sum()}")

    # TODO (S5-2) : configurer l'URI de tracking et l'experience
    # TODO (S5-3) : ouvrir un run MLflow (with mlflow.start_run())

    # --- Entrainement ---
    model = build_model(c=c, max_iter=max_iter)
    print(f"\nEntrainement LogisticRegression (C={c}, max_iter={max_iter})...")
    model.fit(x_train, y_train)

    # --- Evaluation ---
    proba = model.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "f1":       float(f1_score(y_test, preds)),
        "roc_auc":  float(roc_auc_score(y_test, proba)),
    }

    print(f"\n{'─'*50}")
    print(f"  f1       = {metrics['f1']:.4f}")
    print(f"  roc_auc  = {metrics['roc_auc']:.4f}")
    print(f"{'─'*50}")
    print("\nRapport de classification :")
    print(classification_report(y_test, preds, target_names=["Standard/Poor", "Good"]))

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_test, preds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ConfusionMatrixDisplay(cm, display_labels=["Standard/Poor", "Good"]).plot(
        ax=axes[0], colorbar=False
    )
    axes[0].set_title("Matrice de confusion")

    # --- Courbe ROC ---
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlabel("Taux faux positifs")
    axes[1].set_ylabel("Taux vrais positifs")
    axes[1].set_title("Courbe ROC")
    axes[1].legend()

    plt.tight_layout()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = MODEL_DIR / "evaluation.png"
    plt.savefig(fig_path)
    plt.show()
    print(f"\nGraphiques sauvegardes : {fig_path}")

    # TODO (S5-4) : logger les parametres avec mlflow.log_params
    # TODO (S5-5) : logger les metriques avec mlflow.log_metrics
    # TODO (S5-6) : logger le modele avec mlflow.sklearn.log_model
    # TODO (S5-7 bonus) : logger evaluation.png comme artefact MLflow

    # --- Sauvegarde du modele ---
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"Modele sauvegarde    : {model_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Credit Score Classifier")
    parser.add_argument("--c",        type=float, default=1.0,  help="Regularisation LogisticRegression")
    parser.add_argument("--max-iter", type=int,   default=1000, help="Iterations max")
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()

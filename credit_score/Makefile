# ==============================================================================
# Credit Score Classifier - Makefile (vitrine)
# Problematique : predire si un client a un bon score de credit (Good=1 / Other=0)
# Dataset       : ../data/dataset_train.csv (Kaggle - Credit Score Classification)
#
# Ce Makefile delegue l'execution au projet de travail (todo/).
# Le code source est dans : ../mlops-iabd-esgi/todo/mlproject/
# ==============================================================================

TODO := ../mlops-iabd-esgi/todo

# Hyperparametres (surchargeables via CLI : make train C=0.5)
C             ?= 1.0
MAX_ITER      ?= 1000
CV            ?= 5
SCORING       ?= roc_auc
N_TRIALS      ?= 30
API_HOST      ?= 127.0.0.1
API_PORT      ?= 8000
FRONTEND_PORT ?= 8501

CYAN  := $(shell printf '\033[36m')
RESET := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help install sync lock reset-env doctor \
        data train train-optuna train-models \
        mlflow api frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check


# ==============================================================================
# Aide
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
	    {printf "$(CYAN)%-18s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Environnement
# ==============================================================================

install: ## Installe les dependances dans todo/ (uv sync --extra dev)
	$(MAKE) -C $(TODO) install

sync: install ## Alias de install

lock: ## Regenere uv.lock dans todo/
	$(MAKE) -C $(TODO) lock

reset-env: ## Reinitialise completement l'environnement dans todo/
	$(MAKE) -C $(TODO) reset-env

doctor: ## Diagnostique l'environnement (versions des librairies)
	$(MAKE) -C $(TODO) doctor


# ==============================================================================
# Pipeline ML
# ==============================================================================

data: ## Prepare ../data/dataset.csv (nettoyage + binarisation Credit_Score)
	$(MAKE) -C $(TODO) data

train: ## Entraine la baseline LogisticRegression (C=.. MAX_ITER=..)
	$(MAKE) -C $(TODO) train C=$(C) MAX_ITER=$(MAX_ITER)

train-optuna: ## Optimise avec Optuna (N_TRIALS=.. CV=.. SCORING=..)
	$(MAKE) -C $(TODO) train-optuna N_TRIALS=$(N_TRIALS) CV=$(CV) SCORING=$(SCORING)

train-models: ## Compare RF / XGBoost / LightGBM via GridSearchCV + SHAP
	$(MAKE) -C $(TODO) train-models CV=$(CV) SCORING=$(SCORING)

mlflow: ## Demarre le serveur MLflow
	$(MAKE) -C $(TODO) mlflow

api: ## Lance l'API FastAPI (API_HOST / API_PORT)
	$(MAKE) -C $(TODO) api API_HOST=$(API_HOST) API_PORT=$(API_PORT)

frontend: ## Lance le frontend Streamlit (FRONTEND_PORT)
	$(MAKE) -C $(TODO) frontend FRONTEND_PORT=$(FRONTEND_PORT)


# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit l'image d'entrainement
	$(MAKE) -C $(TODO) docker-build

docker-run: ## Entraine en conteneur
	$(MAKE) -C $(TODO) docker-run

docker-up: ## Demarre la stack complete (mlflow + api + frontend)
	$(MAKE) -C $(TODO) docker-up

docker-down: ## Arrete et supprime les conteneurs
	$(MAKE) -C $(TODO) docker-down


# ==============================================================================
# Qualite du code
# ==============================================================================

lint: ## Verifie le style (ruff)
	$(MAKE) -C $(TODO) lint

format: ## Formate le code (ruff)
	$(MAKE) -C $(TODO) format

type: ## Verifie les types (mypy)
	$(MAKE) -C $(TODO) type

test: ## Lance les tests (pytest)
	$(MAKE) -C $(TODO) test

check: lint type test ## Pipeline qualite complete (lint + types + tests)

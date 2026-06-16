# ==============================================================================
# Credit Score Classifier - Makefile
# Problematique : predire si un client a un bon score de credit (Good=1 / Other=0)
# Dataset       : src/data/dataset.csv (Kaggle - Credit Score Classification)
#
# Environnement : venv gere par uv dans ../mlops-iabd-esgi/todo/.venv
# ==============================================================================

VENV        := ../mlops-iabd-esgi/todo/.venv
PYTHON      := $(VENV)/bin/python
TODO        := ../mlops-iabd-esgi/todo
SRC         := src

# Hyperparametres (surchargeables : make train-models CV=5)
CV            ?= 3
SCORING       ?= roc_auc
N_TRIALS      ?= 30
C             ?= 1.0
MAX_ITER      ?= 1000
API_HOST      ?= 127.0.0.1
API_PORT      ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT   ?= 5000

CYAN  := $(shell printf '\033[36m')
RESET := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help install sync lock reset-env doctor \
        mlflow train-models train-optuna train \
        api frontend \
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

install: ## Installe les dependances (uv sync --extra dev dans todo/)
	$(MAKE) -C $(TODO) install

sync: install ## Alias de install

lock: ## Regenere uv.lock dans todo/
	$(MAKE) -C $(TODO) lock

reset-env: ## Reinitialise completement l'environnement
	$(MAKE) -C $(TODO) reset-env

doctor: ## Diagnostique les versions des librairies
	$(MAKE) -C $(TODO) doctor


# ==============================================================================
# MLflow
# ==============================================================================

mlflow: ## Demarre le serveur MLflow sur le port $(MLFLOW_PORT)
	cd $(TODO) && uv run mlflow server \
	    --host 127.0.0.1 --port $(MLFLOW_PORT) \
	    --backend-store-uri sqlite:///mlflow.db \
	    --artifacts-destination ./mlflow-artifacts \
	    --allowed-hosts "*"


# ==============================================================================
# Pipeline ML
# ==============================================================================

train: ## Entraine la baseline LogisticRegression (C=.. MAX_ITER=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train C=$(C) MAX_ITER=$(MAX_ITER)

train-models: ## Compare RF / XGBoost / LightGBM via GridSearchCV + SHAP (CV=.. SCORING=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train_models --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimise avec Optuna TPE (N_TRIALS=.. CV=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train_optuna --n-trials $(N_TRIALS) --cv $(CV)


# ==============================================================================
# Services
# ==============================================================================

api: ## Lance l'API FastAPI (API_HOST / API_PORT)
	$(MAKE) -C $(TODO) api API_HOST=$(API_HOST) API_PORT=$(API_PORT)

frontend: ## Lance le frontend Streamlit (FRONTEND_PORT)
	$(MAKE) -C $(TODO) frontend FRONTEND_PORT=$(FRONTEND_PORT)


# ==============================================================================
# Qualite du code
# ==============================================================================

lint: ## Verifie le style avec ruff
	$(VENV)/bin/ruff check $(SRC)/

format: ## Formate le code avec ruff
	$(VENV)/bin/ruff format $(SRC)/

type: ## Verifie les types avec mypy
	$(VENV)/bin/mypy $(SRC)/

test: ## Lance les tests avec pytest
	PYTHONPATH=$(SRC) $(VENV)/bin/pytest

check: lint type test ## Pipeline qualite complete (lint + types + tests)

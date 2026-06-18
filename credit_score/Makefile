# ==============================================================================
# Credit Score Classifier - Makefile
# Problematique : predire si un client a un bon score de credit (Good=1 / Other=0)
# Dataset       : src/data/dataset.csv (Kaggle - Credit Score Classification)
#
# Usage rapide :
#   make install   -> cree le venv local et installe les dependances
#   make api       -> lance l'API FastAPI sur http://127.0.0.1:8000
#   make test      -> lance les tests
#   make check     -> lint + types + tests
# ==============================================================================

VENV   := .venv
# Utilise python3.13/3.12/3.11/3.10 si disponible, sinon python3
PYTHON_CMD := $(shell \
    command -v python3.13 2>/dev/null || \
    command -v python3.12 2>/dev/null || \
    command -v python3.11 2>/dev/null || \
    command -v python3.10 2>/dev/null || \
    command -v python3 2>/dev/null)
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
SRC    := src

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

COMPOSE := docker compose -f ../docker-compose.yml

.PHONY: help install api mlflow train train-models train-optuna \
        workflow-docker docker-down docker-reset airflow airflow-password \
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

install: ## Cree le venv local et installe toutes les dependances (pip)
	$(PYTHON_CMD) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "$(CYAN)Environnement pret. Copiez .env.example -> .env si besoin.$(RESET)"


# ==============================================================================
# MLflow
# ==============================================================================

mlflow: ## Demarre le serveur MLflow sur le port $(MLFLOW_PORT)
	$(VENV)/bin/mlflow server \
	    --host 127.0.0.1 --port $(MLFLOW_PORT) \
	    --backend-store-uri sqlite:///mlflow.db \
	    --artifacts-destination ./mlflow-artifacts \
	    --allowed-hosts "*"


# ==============================================================================
# Pipeline ML
# ==============================================================================

train: ## Entraine la baseline LogisticRegression (C=.. MAX_ITER=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train --c $(C) --max-iter $(MAX_ITER)

train-models: ## Compare RF / XGBoost / LightGBM via GridSearchCV + SHAP (CV=.. SCORING=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train_models --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimise avec Optuna TPE (N_TRIALS=.. CV=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train_optuna --n-trials $(N_TRIALS) --cv $(CV)


# ==============================================================================
# Services
# ==============================================================================

api: ## Lance l'API FastAPI sur http://$(API_HOST):$(API_PORT)/docs
	PYTHONPATH=$(SRC) $(VENV)/bin/uvicorn api:app --reload \
	    --host $(API_HOST) --port $(API_PORT)

frontend: ## Lance le frontend Streamlit (FRONTEND_PORT)
	$(VENV)/bin/streamlit run frontend/app.py \
	    --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker
# ==============================================================================

workflow-docker: ## Lance la stack complete : mlflow -> train -> api + frontend
	$(COMPOSE) up -d mlflow
	$(COMPOSE) --profile train run --rm train
	$(COMPOSE) up -d api frontend

docker-down: ## Arrete tous les services (conserve les donnees)
	$(COMPOSE) down

docker-reset: ## Arrete tous les services ET efface les volumes
	$(COMPOSE) down -v

docker-train-models: ## Lance RF/XGBoost/LightGBM GridSearchCV dans Docker (logue dans MLflow)
	$(COMPOSE) --profile train-models run --rm train-models

docker-train-optuna: ## Lance Optuna RF/XGBoost/LightGBM dans Docker (logue dans MLflow)
	$(COMPOSE) --profile train-optuna run --rm train-optuna

airflow: ## Demarre Airflow (UI sur http://localhost:8080, login: admin)
	$(COMPOSE) --profile airflow up -d airflow

airflow-password: ## Affiche le mot de passe admin Airflow
	$(COMPOSE) exec airflow cat /opt/airflow/standalone_admin_password.txt


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
	PYTHONPATH=$(SRC) $(VENV)/bin/pytest --tb=short -v

check: lint type test ## Pipeline qualite complete (lint + types + tests)

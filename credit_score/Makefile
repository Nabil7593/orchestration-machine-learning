# ==============================================================================
# Credit Score Classifier — Makefile
# Auteur  : Nabil SARKER — ESGI MLOps 2025
# Dataset : Kaggle Credit Score Classification (100 000 clients, 20 features)
#
# Usage   : make <cible>   (depuis credit_score/)
#           make help      pour la liste complète
# ==============================================================================

VENV   := .venv
PYTHON := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
SRC    := src

PYTHON_CMD := $(shell \
    command -v python3.13 2>/dev/null || \
    command -v python3.12 2>/dev/null || \
    command -v python3.11 2>/dev/null || \
    command -v python3.10 2>/dev/null || \
    command -v python3    2>/dev/null)

# Hyperparametres (surchargeables en ligne de commande)
CV            ?= 3
SCORING       ?= roc_auc
N_TRIALS      ?= 10
C             ?= 1.0
MAX_ITER      ?= 1000
API_HOST      ?= 127.0.0.1
API_PORT      ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT   ?= 5000

CYAN  := $(shell printf '\033[36m')
BOLD  := $(shell printf '\033[1m')
RESET := $(shell printf '\033[0m')

# Raccourci vers docker compose depuis la racine du dépôt
COMPOSE := docker compose -f ../docker-compose.yml

.DEFAULT_GOAL := help

.PHONY: help \
        install \
        api frontend mlflow \
        train train-models train-optuna \
        lint format type test check \
        docker-up docker-down docker-reset docker-logs docker-ps \
        docker-train docker-train-models docker-train-optuna \
        docker-airflow docker-airflow-password docker-airflow-logs \
        ec2-deploy ec2-status


# ==============================================================================
# AIDE
# ==============================================================================

help: ## Affiche cette aide
	@printf "\n$(BOLD)Credit Score Classifier$(RESET) — Nabil SARKER — ESGI MLOps 2025\n\n"
	@printf "$(CYAN)%-22s$(RESET) %s\n" "Cible" "Description"
	@printf "%-22s %s\n" "------" "-----------"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
	    {printf "$(CYAN)%-22s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n"


# ==============================================================================
# ENVIRONNEMENT LOCAL
# ==============================================================================

install: ## Crée le venv et installe toutes les dépendances
	$(PYTHON_CMD) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@printf "\n$(CYAN)Venv prêt. Copiez .env.example -> .env si besoin.$(RESET)\n"


# ==============================================================================
# SERVICES LOCAUX (sans Docker)
# ==============================================================================

mlflow: ## Lance MLflow en local sur le port $(MLFLOW_PORT)
	$(VENV)/bin/mlflow server \
	    --host 127.0.0.1 --port $(MLFLOW_PORT) \
	    --backend-store-uri sqlite:///mlflow.db \
	    --default-artifact-root ./mlflow-artifacts

api: ## Lance l'API FastAPI en local → http://$(API_HOST):$(API_PORT)/docs
	PYTHONPATH=$(SRC) $(VENV)/bin/uvicorn api:app --reload \
	    --host $(API_HOST) --port $(API_PORT)

frontend: ## Lance le frontend Streamlit en local → http://localhost:$(FRONTEND_PORT)
	$(VENV)/bin/streamlit run frontend/app.py \
	    --server.port $(FRONTEND_PORT)


# ==============================================================================
# ENTRAÎNEMENT LOCAL (sans Docker)
# ==============================================================================

train: ## Baseline LogisticRegression (C=.. MAX_ITER=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train --c $(C) --max-iter $(MAX_ITER)

train-models: ## Comparaison RF/XGBoost/LightGBM GridSearchCV (CV=.. SCORING=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train_models --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimisation Optuna TPE (N_TRIALS=.. CV=..)
	PYTHONPATH=$(SRC) $(PYTHON) -m train_optuna --n-trials $(N_TRIALS) --cv $(CV)


# ==============================================================================
# QUALITÉ DU CODE
# ==============================================================================

lint: ## Vérifie le style avec ruff
	$(VENV)/bin/ruff check $(SRC)/

format: ## Formate le code avec ruff
	$(VENV)/bin/ruff format $(SRC)/

type: ## Vérifie les types avec mypy
	$(VENV)/bin/mypy $(SRC)/

test: ## Lance les tests avec pytest
	PYTHONPATH=$(SRC) $(VENV)/bin/pytest --tb=short -v

check: lint type test ## Pipeline qualité complète (lint + types + tests)


# ==============================================================================
# DOCKER — STACK COMPLÈTE
# ==============================================================================

docker-up: ## Lance la stack complète : MLflow + train + API + frontend
	$(COMPOSE) up -d mlflow
	$(COMPOSE) --profile train run --rm train
	$(COMPOSE) up -d api frontend
	@printf "\n$(CYAN)Stack prête :$(RESET)\n"
	@printf "  Frontend : http://localhost:8501\n"
	@printf "  API docs : http://localhost:8000/docs\n"
	@printf "  MLflow   : http://localhost:5001\n"

docker-down: ## Arrête tous les containers (conserve les données)
	$(COMPOSE) down

docker-reset: ## Arrête tous les containers ET supprime les volumes
	$(COMPOSE) down -v

docker-ps: ## Affiche l'état de tous les containers
	$(COMPOSE) ps -a

docker-logs: ## Affiche les logs en temps réel (Ctrl+C pour quitter)
	$(COMPOSE) logs -f


# ==============================================================================
# DOCKER — ENTRAÎNEMENT
# ==============================================================================

docker-train: ## Baseline LogisticRegression dans Docker (logs → MLflow)
	$(COMPOSE) --profile train run --rm train

docker-train-models: ## RF/XGBoost/LightGBM GridSearchCV dans Docker (logs → MLflow)
	$(COMPOSE) --profile train-models run --rm train-models

docker-train-optuna: ## Optuna RF/XGBoost/LightGBM dans Docker (logs → MLflow)
	$(COMPOSE) --profile train-optuna run --rm train-optuna

docker-train-all: docker-train docker-train-models docker-train-optuna ## Lance les 3 entraînements séquentiellement


# ==============================================================================
# DOCKER — AIRFLOW
# ==============================================================================

docker-airflow: ## Démarre Airflow → http://localhost:8080 (login: admin)
	$(COMPOSE) --profile airflow up -d airflow
	@printf "\n$(CYAN)Airflow démarre...$(RESET)\n"
	@printf "  UI       : http://localhost:8080\n"
	@printf "  Password : make docker-airflow-password (après ~30s)\n"

docker-airflow-password: ## Affiche le mot de passe admin Airflow
	$(COMPOSE) exec airflow cat /opt/airflow/standalone_admin_password.txt

docker-airflow-logs: ## Affiche les logs Airflow en temps réel
	$(COMPOSE) logs -f airflow


# ==============================================================================
# EC2 — COMMANDES DE DÉPLOIEMENT
# ==============================================================================

EC2_IP  ?= 35.181.53.42
EC2_KEY ?= ~/Downloads/credit-score-key.pem
EC2_SSH := ssh -i $(EC2_KEY) ubuntu@$(EC2_IP)

ec2-deploy: ## Déploie la stack complète sur EC2 (git pull + rebuild)
	$(EC2_SSH) "cd ~/orchestration-machine-learning && git pull \
	    && docker compose up -d mlflow \
	    && docker compose --profile train run --rm train \
	    && docker compose up -d --build api frontend"
	@printf "\n$(CYAN)Déployé sur EC2 :$(RESET)\n"
	@printf "  Frontend : http://$(EC2_IP):8501\n"
	@printf "  API docs : http://$(EC2_IP):8000/docs\n"
	@printf "  MLflow   : http://$(EC2_IP):5001\n"
	@printf "  Airflow  : http://$(EC2_IP):8080\n"

ec2-status: ## Affiche l'état des containers sur EC2
	$(EC2_SSH) "cd ~/orchestration-machine-learning && docker compose ps -a"

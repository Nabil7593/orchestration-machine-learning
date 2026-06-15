# ==============================================================================
# Credit Score Classifier - Makefile
# Problematique : predire si un client a un bon score de credit (Good=1 / Other=0)
# Dataset       : data/dataset_train.csv (Kaggle - Credit Score Classification)
# Environnement : uv + Python 3.13 (pyproject.toml)
# ==============================================================================

SHELL         := /bin/sh
PYTHON        := uv run python
RUN           := uv run
VENV_DIR      := .venv
PYTHONPATH    ?= .
export PYTHONPATH

# Chemins
DATA_RAW      := ../data/dataset_train.csv
DATA_CLEAN    := ../data/dataset.csv
MODEL_DIR     := models

# Serveurs
API_HOST      ?= 127.0.0.1
API_PORT      ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT   := 5000

# Hyperparametres (surchargeables via CLI)
C             ?= 1.0
MAX_ITER      ?= 1000
CV            ?= 5
SCORING       ?= roc_auc
N_TRIALS      ?= 30

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv install sync lock reset-env doctor \
        data train train-models train-optuna mlflow api frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check


# ==============================================================================
# Aide
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
	    {printf "$(CYAN)%-18s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Environnement (uv)
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
	    echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
	    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
	    echo "  source $$HOME/.local/bin/env"; \
	    exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
	    echo "$(RED)[ERREUR] Virtualenv manquant$(RESET) - lance : make install"; \
	    exit 1; \
	}

install: check-uv ## Installe les dependances (uv sync --extra dev)
	@echo "$(YELLOW)>> Installation des dependances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"

sync: install ## Alias de install

lock: check-uv ## Regenere uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Generation du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock mis a jour$(RESET)"

reset-env: check-uv ## Reinitialise completement l'environnement
	@echo "$(YELLOW)>> Suppression de .venv et uv.lock...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recree$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement
	@echo "$(CYAN)>> Versions$(RESET)"
	@uv --version
	@$(PYTHON) --version
	@$(PYTHON) -c "import sklearn, xgboost, lightgbm, mlflow, optuna, shap; \
	    print(f'  sklearn={sklearn.__version__}  xgb={xgboost.__version__}  lgbm={lightgbm.__version__}  mlflow={mlflow.__version__}')"
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"


# ==============================================================================
# Pipeline ML
# ==============================================================================

data: check-venv ## Prepare dataset.csv depuis dataset_train.csv (nettoyage + binarisation)
	@echo "$(YELLOW)>> Preparation du dataset Credit Score...$(RESET)"
	$(PYTHON) -m mlproject.prepare_data
	@echo "$(GREEN)[OK] data/dataset.csv genere$(RESET)"

train: check-venv ## Entraine la baseline LogisticRegression + suivi MLflow (C=.. MAX_ITER=..)
	@echo "$(YELLOW)>> Entrainement baseline (C=$(C), max_iter=$(MAX_ITER))...$(RESET)"
	$(PYTHON) -m mlproject.train --c $(C) --max-iter $(MAX_ITER)

train-optuna: check-venv ## Optimise LR/RF/XGB/LGB avec Optuna (N_TRIALS=.. CV=.. SCORING=..)
	@echo "$(YELLOW)>> Optimisation Optuna (n_trials=$(N_TRIALS), cv=$(CV), scoring=$(SCORING))...$(RESET)"
	$(PYTHON) -m mlproject.train_optuna --n-trials $(N_TRIALS) --cv $(CV) --scoring $(SCORING)

train-models: check-venv ## Compare RF / XGBoost / LightGBM via GridSearchCV + SHAP (CV=.. SCORING=..)
	@echo "$(YELLOW)>> Comparaison de modeles (cv=$(CV), scoring=$(SCORING))...$(RESET)"
	$(PYTHON) -m mlproject.train_models --cv $(CV) --scoring $(SCORING)

mlflow: check-venv ## Demarre le serveur MLflow (docker compose)
	docker compose -f docker-compose.yml up -d mlflow
	@echo "$(GREEN)[OK] MLflow disponible sur http://127.0.0.1:$(MLFLOW_PORT)$(RESET)"

api: check-venv ## Lance l'API FastAPI en mode rechargement (API_HOST / API_PORT)
	$(RUN) uvicorn mlproject.api:app --reload --host $(API_HOST) --port $(API_PORT)

frontend: check-venv ## Lance le frontend Streamlit (FRONTEND_PORT)
	$(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit l'image d'entrainement
	docker build -f docker/Dockerfile.train -t credit-score-train .

docker-run: ## Entraine en conteneur (monte models/ depuis l'hote)
	docker run --rm \
	    -v "$(CURDIR)/../data:/app/data:ro" \
	    -v "$(CURDIR)/$(MODEL_DIR):/app/models" \
	    credit-score-train

docker-up: ## Demarre la stack complete (mlflow + api + frontend)
	docker compose -f docker-compose.yml up -d --build mlflow api frontend
	@echo "$(GREEN)[OK] Stack demarree$(RESET)"
	@echo "  MLflow   : http://127.0.0.1:$(MLFLOW_PORT)"
	@echo "  API      : http://127.0.0.1:$(API_PORT)/docs"
	@echo "  Frontend : http://127.0.0.1:$(FRONTEND_PORT)"

docker-down: ## Arrete et supprime les conteneurs (conserve les volumes)
	docker compose -f docker-compose.yml down


# ==============================================================================
# Qualite du code
# ==============================================================================

lint: check-venv ## Verifie le style (ruff)
	$(RUN) ruff check mlproject

format: check-venv ## Formate le code (ruff)
	$(RUN) ruff format mlproject

type: check-venv ## Verifie les types (mypy)
	$(RUN) mypy mlproject

test: check-venv ## Lance les tests (pytest)
	$(RUN) pytest

check: lint type test ## Pipeline qualite complete (lint + types + tests)

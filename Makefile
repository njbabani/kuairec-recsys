.DEFAULT_GOAL := help
.PHONY: help setup data features eda results experiments demo-data demo mlflow-ui sweep-als pipeline test lint format check

help: ## List available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create the Python 3.12 environment from uv.lock
	uv sync --locked

data: ## Download KuaiRec, build validated tables and labeled splits (DVC)
	uv run dvc repro split

features: ## Build model-ready user and video feature tables (DVC)
	uv run dvc repro features

eda: ## Re-run the EDA notebook (after `make data`) and refresh reports/figures
	uv run jupytext --to notebook --execute notebooks/01_eda.py

results: ## Re-run the results notebooks (after `make experiments`)
	uv run jupytext --to notebook --execute notebooks/02_baselines.py
	uv run jupytext --to notebook --execute notebooks/03_two_tower.py
	uv run jupytext --to notebook --execute notebooks/04_reranker.py
	uv run jupytext --to notebook --execute notebooks/05_ab_testing.py

experiments: ## Model searches, re-ranker, final test evaluation and A/B tests (MLflow + W&B)
	uv run dvc repro als_search two_tower_search leaderboard retrieval reranker final_evaluation ab_test

demo-data: ## Build the Streamlit demo's tables (DVC; runs any missing pipeline stage first)
	uv run dvc repro demo_data

demo: ## Launch the Streamlit demo at http://localhost:8501
	uv run streamlit run app/streamlit_app.py

mlflow-ui: ## Browse tracked runs at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

sweep-als: ## Create a W&B Bayesian sweep for ALS (needs `wandb login` first)
	uv run wandb sweep sweeps/als.yaml

pipeline: ## Reproduce every DVC stage whose inputs changed
	uv run dvc repro

test: ## Run the test suite with coverage (fails under 80%)
	uv run pytest --cov

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

check: lint test ## Everything CI runs
	uv run dvc dag > /dev/null

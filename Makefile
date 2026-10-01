.DEFAULT_GOAL := help
.PHONY: help setup data eda pipeline test lint format check

help: ## List available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup: ## Create the Python 3.12 environment from uv.lock
	uv sync --locked

data: ## Download KuaiRec, build validated tables and labeled splits (DVC)
	uv run dvc repro split

eda: ## Re-run the EDA notebook (after `make data`) and refresh reports/figures
	uv run jupytext --to notebook --execute notebooks/01_eda.py

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

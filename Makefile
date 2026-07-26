.DEFAULT_GOAL := help
PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin
BENCH ?= conf/benchmarks/manipulation_v1.yaml

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install with dev tools
	$(BIN)/pip install -e '.[dev]'

.PHONY: install-all
install-all: $(BIN)/python ## Install with torch, for checkpoint evaluation
	$(BIN)/pip install -e '.[dev,torch]'

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: lint
lint: ## Lint and format-check
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

.PHONY: fmt
fmt: ## Auto-format
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

.PHONY: baselines
baselines: ## Run every reference policy, to confirm the suite discriminates
	@for p in zero random scripted scripted+noise:0.05 scripted+noise:0.12; do \
		printf "%-24s " "$$p"; \
		$(BIN)/erl-evals run "$$p" --benchmark $(BENCH) 2>&1 | grep overall; \
	done

.PHONY: demo-gate
demo-gate: ## Show the gate catching an injected regression
	$(BIN)/erl-evals run scripted --benchmark $(BENCH) --tag baseline
	$(BIN)/erl-evals run scripted+noise:0.12 --benchmark $(BENCH) --tag candidate
	-$(BIN)/erl-evals compare \
		results/manipulation_v1__baseline.json \
		results/manipulation_v1__candidate.json

.PHONY: clean
clean: ## Remove results and reports
	rm -rf results reports registry
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV) .pytest_cache .ruff_cache

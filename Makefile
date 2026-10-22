# All targets run inside the repository's own .venv (created by uv).
UV ?= uv
RUN = $(UV) run --frozen

.PHONY: setup lint format typecheck test reproduce report serve clean

setup:
	$(UV) venv --python 3.13 .venv --allow-existing
	$(UV) sync --frozen
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example: set TRADEQ_DATA_DIR")

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format:
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest --cov --cov-report=term

reproduce:
	$(RUN) tsfm-arena reproduce

report:
	$(RUN) tsfm-arena report

serve:
	$(RUN) tsfm-arena serve

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov

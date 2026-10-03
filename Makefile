export UV_PROJECT_ENVIRONMENT ?= .venv

setup: 
	uv sync --extra ml --group dev

check: 
	uv run ruff check . && uv run ruff format --check . && uv run mypy perception scripts 

ui: 
	uv run --extra ml --extra ui perception-ui

cov:
	uv run pytest -m "not model" --cov --cov-report=term-missing --cov-report=html -q
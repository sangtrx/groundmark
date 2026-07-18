.PHONY: install test lint run worker docker-up docker-down

install:
	python -m pip install -e '.[dev]'

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest

lint:
	ruff check src tests

run:
	groundmark serve --port 8000

worker:
	groundmark-worker

docker-up:
	docker compose up --build

docker-down:
	docker compose down

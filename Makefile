.PHONY: bootstrap test simulate run run-local mlflow clean

bootstrap:
	UV_PROJECT_ENVIRONMENT=.venv uv sync

test:
	uv run pytest

simulate:
	uv run python data/simulate_stream.py

# Drift detection only — no API calls, no W&B
run-local:
	uv run python run_monitor.py --no-quality --no-wandb

# Full run: drift + quality scoring + W&B dashboard
run:
	uv run python run_monitor.py

mlflow:
	uv run mlflow ui

clean:
	rm -rf data/batches/ mlruns/ mlflow.db wandb/

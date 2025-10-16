.PHONY: install simulate run run-local mlflow clean

install:
	pip install -r requirements.txt

simulate:
	python data/simulate_stream.py

# Drift detection only — no API calls, no W&B
run-local:
	python run_monitor.py --no-quality --no-wandb

# Full run: drift + quality scoring + W&B dashboard
run:
	python run_monitor.py

mlflow:
	mlflow ui

clean:
	rm -rf data/batches/ mlruns/ mlflow.db wandb/

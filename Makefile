.PHONY: install data features train evaluate serve prefect-flow lint lint-fix test drift-report

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

data:
	python -m src.data.ingest

features:
	python -m src.features.build_features

train:
	python -m src.models.train

evaluate:
ifndef RUN_ID
	$(error Please provide RUN_ID, e.g. `make evaluate RUN_ID=<mlflow-run-id>`)
endif
	python -m src.models.evaluate --run-id $(RUN_ID)

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

prefect-flow:
	python -m src.workflows.prefect_flow

lint:
	ruff check src tests
	black --check src tests

lint-fix:
	ruff check src tests --fix
	black src tests

test:
	python -m pytest

drift-report:
ifdef MODEL_URI
	python scripts/generate_drift_report.py --model-uri $(MODEL_URI)
else
	python scripts/generate_drift_report.py
endif

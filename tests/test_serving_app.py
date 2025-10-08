import importlib
import os

from fastapi.testclient import TestClient

import src.serving.app as serving_app


def reload_app_with_skip() -> None:
    os.environ["SKIP_MODEL_LOAD"] = "1"
    module = importlib.reload(serving_app)
    globals()["serving_app"] = module


def test_metrics_endpoint():
    reload_app_with_skip()
    client = TestClient(serving_app.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "finance_math_request_total" in response.text


def test_predict_returns_503_when_model_missing():
    reload_app_with_skip()
    client = TestClient(serving_app.app)
    response = client.post("/predict", json={"question": "test", "tables_markdown": []})
    assert response.status_code == 503

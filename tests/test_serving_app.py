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
    assert "X-Request-ID" in response.headers
    assert "request_id" in response.json()["detail"]


def test_predict_success_returns_request_id(monkeypatch):
    os.environ.pop("SKIP_MODEL_LOAD", None)

    def _fake_load_predictor():
        class _DummyPredictor:
            def predict(self, _):
                return 3.14

        return _DummyPredictor()

    module = importlib.reload(serving_app)
    globals()["serving_app"] = module
    monkeypatch.setattr(module, "load_predictor", _fake_load_predictor)
    module.predictor = None

    client = TestClient(module.app)
    response = client.post("/predict", json={"question": "ok", "tables_markdown": []})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 3.14
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]

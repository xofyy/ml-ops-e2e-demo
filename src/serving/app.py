from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from src.serving.predict import FinanceMathRequest, load_predictor, Predictor

app = FastAPI(
    title="FinanceMath Inference API",
    version="0.1.0",
    description="Serve regression predictions for FinanceMath questions.",
)

predictor: Predictor | None = None
LOGGER = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "finance_math_request_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "http_status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "finance_math_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    registry=REGISTRY,
)
PREDICTION_COUNTER = Counter(
    "finance_math_predictions_total",
    "Total number of predictions served",
    registry=REGISTRY,
)


@app.on_event("startup")
def startup_event() -> None:
    global predictor
    if os.environ.get("SKIP_MODEL_LOAD") == "1":
        LOGGER.warning("Skipping model load because SKIP_MODEL_LOAD=1")
        predictor = None
    else:
        predictor = load_predictor()


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed = time.perf_counter() - start
        status_code = response.status_code if response is not None else 500
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.url.path, http_status=status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method, endpoint=request.url.path
        ).observe(elapsed)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: FinanceMathRequest) -> dict[str, float]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    result = predictor.predict(request)
    PREDICTION_COUNTER.inc()
    return {"prediction": result}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

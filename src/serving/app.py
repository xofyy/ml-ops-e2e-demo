from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    generate_latest,
    Summary,
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
    ["method", "endpoint", "status_code"],
    registry=REGISTRY,
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
REQUEST_LATENCY_SUMMARY = Summary(
    "finance_math_request_latency_quantiles_seconds",
    "Request latency summary for p50/p90/p99 estimation",
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
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response: Response | None = None
    try:
        response = await call_next(request)
    finally:
        elapsed = time.perf_counter() - start
        status_code = response.status_code if response is not None else 500
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            http_status=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(status_code),
        ).observe(elapsed)
        REQUEST_LATENCY_SUMMARY.labels(
            method=request.method, endpoint=request.url.path
        ).observe(elapsed)
        if response is not None:
            response.headers["X-Request-ID"] = request_id
    if response is None:
        response = Response(status_code=500)
        response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: FinanceMathRequest, http_request: Request) -> dict[str, float | str]:
    global predictor  # noqa: PLW0603
    if predictor is None:
        request_id = getattr(http_request.state, "request_id", None)
        if os.environ.get("SKIP_MODEL_LOAD") == "1":
            detail = "Model not loaded yet."
            if request_id:
                detail = f"{detail} (request_id={request_id})"
            raise HTTPException(status_code=503, detail=detail)
        try:
            predictor = load_predictor()
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Failed to load predictor lazily: %s", exc)
            detail = "Model load failed."
            if request_id:
                detail = f"{detail} (request_id={request_id})"
            raise HTTPException(status_code=503, detail=detail) from exc
    result = predictor.predict(request)
    PREDICTION_COUNTER.inc()
    payload: dict[str, float | str] = {"prediction": result}
    request_id = getattr(http_request.state, "request_id", None)
    if request_id:
        payload["request_id"] = request_id
    return payload


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

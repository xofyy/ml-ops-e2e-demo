from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.serving.predict import FinanceMathRequest, load_predictor, Predictor

app = FastAPI(
    title="FinanceMath Inference API",
    version="0.1.0",
    description="Serve regression predictions for FinanceMath questions.",
)

predictor: Predictor | None = None


@app.on_event("startup")
def startup_event() -> None:
    global predictor
    predictor = load_predictor()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: FinanceMathRequest) -> dict[str, float]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    result = predictor.predict(request)
    return {"prediction": result}

from __future__ import annotations

import argparse
from pathlib import Path
from inspect import signature

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report
from mlflow import artifacts
from sklearn import metrics


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type for {path}")


def ensure_required_columns(df: pd.DataFrame, prediction_column: str) -> None:
    missing = {col for col in ("target", prediction_column) if col not in df.columns}
    if missing:
        raise ValueError(
            f"DataFrame must contain columns {missing}. "
            "Provide a dataset with predictions or supply --model-uri to compute them."
        )


def ensure_predictions(
    df: pd.DataFrame, model_uri: str, prediction_column: str
) -> pd.DataFrame:
    if prediction_column in df.columns:
        return df

    model = mlflow.pyfunc.load_model(model_uri)
    feature_columns = load_feature_columns(model_uri)

    features = df.drop(
        columns=[c for c in ("target", prediction_column, "question_id") if c in df],
        errors="ignore",
    )
    features = features.reindex(columns=feature_columns, fill_value=0.0)

    predictions = model.predict(features)
    result = df.copy()
    result[prediction_column] = predictions
    return result


def load_feature_columns(model_uri: str) -> list[str]:
    if model_uri.startswith("runs:/"):
        remainder = model_uri.split("runs:/", 1)[1]
        run_id = remainder.split("/", 1)[0]
        columns_uri = f"runs:/{run_id}/feature_columns.json"
    else:
        columns_uri = f"{model_uri}/feature_columns.json"

    metadata = artifacts.load_dict(columns_uri)
    return metadata["feature_columns"]


def ensure_mean_squared_error_supports_squared() -> None:
    sig = signature(metrics.mean_squared_error)
    if "squared" in sig.parameters:
        return

    from sklearn.metrics import root_mean_squared_error

    original = metrics.mean_squared_error

    def patched_mean_squared_error(
        y_true,
        y_pred,
        *,
        sample_weight=None,
        multioutput="uniform_average",
        squared=True,
    ):
        if squared:
            return original(
                y_true,
                y_pred,
                sample_weight=sample_weight,
                multioutput=multioutput,
            )
        return root_mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )

    metrics.mean_squared_error = patched_mean_squared_error

    try:
        import sklearn.metrics._regression as regression_module  # type: ignore[attr-defined]

        regression_module.mean_squared_error = patched_mean_squared_error  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        import evidently.metrics.regression_performance.regression_quality as regression_quality

        regression_quality.mean_squared_error = patched_mean_squared_error  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> None:
    ensure_mean_squared_error_supports_squared()

    parser = argparse.ArgumentParser(
        description="Generate Evidently regression drift report."
    )
    parser.add_argument(
        "--reference", default="data/processed/finance_math_features.parquet"
    )
    parser.add_argument("--current", required=False)
    parser.add_argument("--output", default="data/processed/evidently_report.html")
    parser.add_argument(
        "--model-uri", help="Optional MLflow model URI to compute predictions."
    )
    parser.add_argument(
        "--registry-uri",
        default="file:./mlruns",
        help="MLflow tracking/registry URI (used when --model-uri is provided).",
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Column name containing predictions (created when --model-uri is used).",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference)
    current_path = Path(args.current) if args.current else reference_path
    output_path = Path(args.output)

    reference_df = load_dataframe(reference_path)
    current_df = load_dataframe(current_path)

    if args.model_uri:
        mlflow.set_tracking_uri(args.registry_uri)
        reference_df = ensure_predictions(
            reference_df, args.model_uri, args.prediction_column
        )
        current_df = ensure_predictions(
            current_df, args.model_uri, args.prediction_column
        )

    ensure_required_columns(reference_df, args.prediction_column)
    ensure_required_columns(current_df, args.prediction_column)

    report = Report(metrics=[RegressionPreset(), DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    print(f"Evidently report saved to {output_path}")


if __name__ == "__main__":
    main()

"""Evaluate the frozen production inference pipeline on known historical rows.

Heavy filtering/sampling stays in DuckDB. Selected rows are passed through
``inference.engine.predict_proposals`` so feature generation and predictions
are identical to production. This script never trains or modifies artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(PROJECT_DIR))
from inference import engine

GROUPS = ["1-10", "11-20", "21-30", "31-40", "41-60", "61-100", "101-300"]
LABELS = ["CLEAR_FAILURE", "WEAK_DEMAND", "MODERATE_DEMAND", "STRONG_DEMAND", "CAPACITY_PRESSURE"]
MODELS = {
    "weekday_baseline": "weekday_baseline_prediction",
    "v4_1_regression": "v4_1_prediction",
    "v4_2_hybrid": "v4_2_hybrid_prediction",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2025-07-01", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-01-01", help="Exclusive YYYY-MM-DD")
    parser.add_argument("--max-rows", type=int, default=10_000, help="0 evaluates the full period")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--batch-days",
        type=int,
        default=7,
        help="Maximum date span per production-engine call (bounds DuckDB memory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "inference_error_analysis",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    start, end = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    if start <= date.fromisoformat(engine.HISTORY_START):
        raise ValueError(f"start date must be later than {engine.HISTORY_START}")
    if end <= start:
        raise ValueError("end date must be later than start date")
    if args.max_rows < 0:
        raise ValueError("max rows cannot be negative")
    if args.batch_days < 1:
        raise ValueError("batch days must be at least 1")


def historical_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, int]:
    where = """
        SEFER_TARIHI >= CAST(? AS DATE) AND SEFER_TARIHI < CAST(? AS DATE)
        AND target BETWEEN 1 AND 300 AND canonical_guzergah_id IS NOT NULL
    """
    params: list[object] = [args.start_date, args.end_date]
    suffix = ""
    if args.max_rows:
        suffix = "ORDER BY HASH(SEFER_ID, ?) LIMIT ?"
        params += [args.sample_seed, args.max_rows]
    with duckdb.connect(str(engine.DB_PATH), read_only=True) as conn:
        count = conn.execute(f"SELECT COUNT(*) FROM model_data_base WHERE {where}", params[:2]).fetchone()[0]
        frame = conn.execute(
            f"""
            SELECT SEFER_ID, FIRMA_ID, GUZERGAH_KODU,
                   STRFTIME(SEFER_TARIHI, '%Y-%m-%d') AS SEFER_TARIHI,
                   CAST(SEFER_SAATI AS VARCHAR) AS SEFER_SAATI,
                   target AS actual
            FROM model_data_base WHERE {where} {suffix}
            """,
            params,
        ).fetchdf()
    if frame.empty:
        raise ValueError("no eligible historical rows in the requested period")
    return frame, int(count)


def regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = predicted - actual
    return {
        "rows": len(actual),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": math.sqrt(mean_squared_error(actual, predicted)),
        "bias": residual.mean(),
        "median_absolute_error": np.median(np.abs(residual)),
        "actual_mean": actual.mean(),
        "predicted_mean": predicted.mean(),
    }


def add_actual_labels(frame: pd.DataFrame) -> None:
    frame["passenger_group"] = pd.cut(
        frame.actual, [0, 10, 20, 30, 40, 60, 100, 300], labels=GROUPS, include_lowest=True
    )
    labels = np.select(
        [frame.actual >= 43, frame.actual >= 30, frame.actual >= 20, frame.actual >= 10],
        ["CAPACITY_PRESSURE", "STRONG_DEMAND", "MODERATE_DEMAND", "WEAK_DEMAND"],
        default="CLEAR_FAILURE",
    )
    frame["actual_demand_label"] = pd.Categorical(labels, LABELS, ordered=True)


def tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall, grouped, classifiers = [], [], []
    actual = frame.actual.to_numpy(float)
    for model, column in MODELS.items():
        overall.append({"model": model, **regression(actual, frame[column].to_numpy(float))})
        for group in GROUPS:
            part = frame[frame.passenger_group == group]
            if not part.empty:
                grouped.append(
                    {"passenger_group": group, "model": model,
                     **regression(part.actual.to_numpy(float), part[column].to_numpy(float))}
                )
    for threshold in engine.THRESHOLDS:
        truth = (frame.actual.to_numpy() >= threshold).astype("int8")
        probability = frame[f"probability_ge_{threshold}"].to_numpy(float)
        auc = roc_auc_score(truth, probability) if np.unique(truth).size == 2 else np.nan
        for kind, column in [
            ("raw_classifier", f"v4_4_raw_decision_ge_{threshold}"),
            ("production_mixed", f"mixed_decision_ge_{threshold}"),
        ]:
            pred = frame[column].to_numpy("int8")
            tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
            classifiers.append({
                "threshold": threshold, "decision_type": kind, "rows": len(frame),
                "positive_rate": truth.mean(), "accuracy": accuracy_score(truth, pred),
                "precision": precision_score(truth, pred, zero_division=0),
                "recall": recall_score(truth, pred, zero_division=0),
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "f1": f1_score(truth, pred, zero_division=0), "roc_auc_probability": auc,
                "brier_score_probability": brier_score_loss(truth, probability),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            })
    return pd.DataFrame(overall), pd.DataFrame(grouped), pd.DataFrame(classifiers)


def plots(frame: pd.DataFrame, grouped: pd.DataFrame, output: Path) -> None:
    actual = frame.actual.to_numpy(float)
    predicted = frame.v4_2_hybrid_prediction.to_numpy(float)
    residual = predicted - actual

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.hexbin(actual, predicted, gridsize=55, bins="log", mincnt=1, cmap="viridis")
    limit = max(actual.max(), predicted.max())
    ax.plot([0, limit], [0, limit], "--", color="crimson", label="Perfect prediction")
    ax.set(title="Actual vs predicted", xlabel="Actual passengers", ylabel="v4.2 hybrid prediction")
    ax.legend(); fig.colorbar(image, ax=ax, label="log10(rows)"); fig.tight_layout()
    fig.savefig(output / "actual_vs_predicted.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.hexbin(predicted, residual, gridsize=55, bins="log", mincnt=1, cmap="magma")
    ax.axhline(0, ls="--", color="black")
    ax.set(title="Residuals vs prediction", xlabel="Prediction", ylabel="Residual (predicted - actual)")
    fig.colorbar(image, ax=ax, label="log10(rows)"); fig.tight_layout()
    fig.savefig(output / "residuals_vs_predicted.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(residual, bins=80, color="#3267a8")
    axes[0].axvline(0, ls="--", color="black")
    axes[0].set(title="Residual distribution", xlabel="Residual", ylabel="Rows")
    axes[1].hist(np.abs(residual), bins=80, color="#d26a32")
    axes[1].set(title="Absolute-error distribution", xlabel="Absolute error", ylabel="Rows")
    fig.tight_layout(); fig.savefig(output / "error_distributions.png", dpi=160); plt.close(fig)

    data = grouped[grouped.model == "v4_2_hybrid"].set_index("passenger_group").reindex(GROUPS)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    axes[0].bar(data.index, data.mae, color="#3267a8")
    axes[0].set(title="v4.2 MAE by actual passenger group", ylabel="MAE")
    axes[1].bar(data.index, data.bias, color=np.where(data.bias >= 0, "#d26a32", "#4c956c"))
    axes[1].axhline(0, color="black"); axes[1].set(title="v4.2 bias by group", ylabel="Bias")
    fig.tight_layout(); fig.savefig(output / "mae_bias_by_passenger_group.png", dpi=160); plt.close(fig)

    for kind, prefix, filename in [
        ("Raw v4.4", "v4_4_raw_decision", "classifier_raw_confusion_matrices.png"),
        ("Production mixed", "mixed_decision", "production_threshold_confusion_matrices.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, threshold in zip(axes.flat, engine.THRESHOLDS, strict=True):
            truth = (frame.actual >= threshold).astype("int8")
            matrix = confusion_matrix(truth, frame[f"{prefix}_ge_{threshold}"], labels=[0, 1])
            ConfusionMatrixDisplay(matrix, display_labels=[f"<{threshold}", f"≥{threshold}"]).plot(
                ax=ax, cmap="Blues", colorbar=False, values_format=",d"
            )
            ax.set_title(f"{kind}: threshold {threshold}")
        fig.tight_layout(); fig.savefig(output / filename, dpi=160); plt.close(fig)

    matrix = confusion_matrix(frame.actual_demand_label.astype(str), frame.mixed_demand_label, labels=LABELS)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=["<10", "10-19", "20-29", "30-42", "43+"]).plot(
        ax=ax, cmap="Purples", colorbar=False, values_format=",d"
    )
    ax.set_title("Production demand-category confusion matrix")
    fig.tight_layout(); fig.savefig(output / "demand_category_confusion_matrix.png", dpi=160); plt.close(fig)


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return None if not np.isfinite(value) else float(value)
    return value


def predict_date_batches(
    history: pd.DataFrame,
    artifacts: engine.FrozenArtifacts,
    batch_days: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Call the unchanged production engine in bounded chronological batches."""

    work = history.copy()
    work["_analysis_order"] = np.arange(len(work))
    dates = pd.to_datetime(work["SEFER_TARIHI"])
    origin = dates.min()
    work["_date_batch"] = ((dates - origin).dt.days // batch_days).astype(int)
    outputs: list[pd.DataFrame] = []
    total_timings: dict[str, float] = {}
    groups = list(work.groupby("_date_batch", sort=True))
    for number, (_, batch) in enumerate(groups, start=1):
        batch_timings: dict[str, float] = {}
        print(
            f"\nDate batch {number}/{len(groups)}: {batch.SEFER_TARIHI.min()} "
            f"through {batch.SEFER_TARIHI.max()} ({len(batch):,} rows)"
        )
        prediction = engine.predict_proposals(
            batch[engine.INPUT_COLUMNS],
            artifacts=artifacts,
            timings=batch_timings,
            log=lambda message: print(f"[inference] {message}", flush=True),
        )
        prediction["_analysis_order"] = batch["_analysis_order"].to_numpy()
        outputs.append(prediction)
        for key, value in batch_timings.items():
            total_timings[key] = total_timings.get(key, 0.0) + float(value)
    combined = pd.concat(outputs, ignore_index=True).sort_values("_analysis_order")
    return combined.drop(columns="_analysis_order").reset_index(drop=True), total_timings


def main() -> None:
    args = arguments(); validate(args)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    history, eligible = historical_rows(args)
    print(f"Selected {len(history):,} of {eligible:,} eligible rows.")
    artifacts = engine.load_frozen_artifacts()
    predictions, timings = predict_date_batches(history, artifacts, args.batch_days)
    frame = pd.concat([history[["SEFER_ID", "actual"]].reset_index(drop=True),
                       predictions.reset_index(drop=True)], axis=1)
    add_actual_labels(frame)
    frame["v4_2_residual"] = frame.v4_2_hybrid_prediction - frame.actual
    frame["v4_2_absolute_error"] = frame.v4_2_residual.abs()
    overall, grouped, classifiers = tables(frame)
    category = confusion_matrix(frame.actual_demand_label.astype(str), frame.mixed_demand_label, labels=LABELS)

    overall.to_csv(output / "overall_regression_metrics.csv", index=False)
    grouped.to_csv(output / "passenger_group_metrics.csv", index=False)
    classifiers.to_csv(output / "classifier_metrics.csv", index=False)
    pd.DataFrame(category, index=LABELS, columns=LABELS).to_csv(output / "demand_category_confusion_matrix.csv")
    frame.to_csv(output / "historical_predictions.csv", index=False)
    plots(frame, grouped, output)

    summary = {
        "period": {"start_inclusive": args.start_date, "end_exclusive": args.end_date,
                   "eligible_rows": eligible, "evaluated_rows": len(frame),
                   "sample_seed": args.sample_seed, "batch_days": args.batch_days},
        "versions": {"numeric": "v4.2 hybrid / frozen v4.1 regression",
                     "classifiers": "v4.4 >=10, >=20, >=30, >=43",
                     "classifier_variants": artifacts.classifier_variants},
        "overall": overall.set_index("model").to_dict("index"),
        "category_accuracy": accuracy_score(frame.actual_demand_label.astype(str), frame.mixed_demand_label),
        "monotonicity_violation_rate": frame.classifier_monotonicity_violation.mean(),
        "monotonicity_correction_rate": frame.classifier_monotonicity_correction_applied.mean(),
        "reliability_counts": frame.prediction_reliability.value_counts().to_dict(),
        "inference_timings_seconds": timings, "elapsed_seconds": perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(native(summary), indent=2), encoding="utf-8")
    print("\n" + overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()

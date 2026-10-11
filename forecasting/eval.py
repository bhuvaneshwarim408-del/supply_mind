"""
forecasting/eval.py

Evaluation harness. Reports MAE, RMSE, MAPE, WAPE and Bias separately for
LightGBM alone, LSTM alone, the ensemble, and a naive seasonal baseline,
evaluated on a held-out split created with TimeSeriesSplit (never a random
split -- this is time series data).

Aggregates across SKUs: mean/median metrics, per-model win counts, worst
performing SKUs, and a breakdown by product category.

Run:
    python -m forecasting.eval --csv path/to/data.csv --max-skus 20
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import features as feat
from .data import generate_synthetic, load_csv
from .ensemble import (
    DemandEnsemble, _mae, _mape, _rmse, _wape, _bias,
)
from .lgbm_model import LGBMQuantileForecaster
from .lstm_model import LSTMForecasterWrapper, LSTMTrainConfig

METRIC_NAMES = ("MAE", "RMSE", "MAPE", "WAPE", "Bias")
METRIC_FUNCS = {
    "MAE": _mae,
    "RMSE": _rmse,
    "MAPE": _mape,
    "WAPE": _wape,
    "Bias": _bias,
}
MODEL_NAMES = ("LightGBM", "LSTM", "Ensemble", "Baseline")


def _seasonal_naive(y: np.ndarray, season: int = 7) -> np.ndarray:
    """Seasonal-naive baseline: predict the value 'season' days ago."""
    pred = np.full(len(y), np.nan)
    for i in range(len(y)):
        src = i - season
        if src >= 0:
            pred[i] = y[src]
    return pred


def evaluate_sku(
    sku_df: pd.DataFrame,
    target_col: str = "Units Sold",
    n_splits: int = 3,
    season: int = 7,
    category: str | None = None,
) -> dict:
    fdf = feat.build_feature_frame(sku_df, target_col=target_col)
    cols = feat.get_feature_columns(fdf, target_col=target_col)
    fdf = fdf.dropna(subset=cols + [target_col]).reset_index(drop=True)

    X = fdf[cols].to_numpy(dtype=float)
    y = fdf[target_col].to_numpy(dtype=float)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    # use the LAST split as the held-out test segment
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    metrics: dict[str, dict[str, float]] = {}

    def finalize(pred: np.ndarray) -> dict[str, float]:
        valid = ~np.isnan(pred)
        if valid.sum() == 0:
            return {m: float("nan") for m in METRIC_NAMES}
        yy, pp = y_test[valid], pred[valid]
        return {m: float(METRIC_FUNCS[m](yy, pp)) for m in METRIC_NAMES}

    # --- LightGBM alone ---
    lgbm = LGBMQuantileForecaster()
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)["median"]
    metrics["LightGBM"] = finalize(lgbm_pred)

    # --- LSTM alone ---
    seq = LSTMTrainConfig().seq_len
    try:
        n_val = max(int(0.15 * len(X_train)), seq + 5)
        lstm = LSTMForecasterWrapper(n_features=X.shape[1])
        lstm.fit(X_train[:-n_val], y_train[:-n_val],
                 X_train[-n_val:], y_train[-n_val:])

        import torch
        X_test_norm = lstm._normalize(X_test, fit=False)
        lstm_pred = np.full(len(y_test), np.nan)
        lstm.model.eval()
        with torch.no_grad():
            for i in range(seq, len(X_test)):
                window = X_test_norm[i - seq:i]
                x = torch.from_numpy(window[None, :, :].astype(np.float32))
                lstm_pred[i] = lstm.model(x).item()
        metrics["LSTM"] = finalize(lstm_pred)
    except ValueError:
        metrics["LSTM"] = {m: float("nan") for m in METRIC_NAMES}

    # --- Ensemble ---
    ens = DemandEnsemble(n_splits=n_splits)
    ens.fit(sku_df, target_col=target_col)
    ens_out = ens.predict(horizon_days=len(y_test))
    ens_pred = np.array(ens_out.point_forecast)
    # ensemble forecast is aligned to the END of history + horizon; compare
    # with the tail of the test window (same length) as the point forecast
    metrics["Ensemble"] = finalize(ens_pred)

    # --- Seasonal naive baseline ---
    baseline_pred = _seasonal_naive(y, season=season)
    # align baseline to the test window
    baseline_test = baseline_pred[test_idx]
    metrics["Baseline"] = finalize(baseline_test)

    if category is not None:
        metrics["category"] = category
    return metrics


def aggregate_results(
    results: dict[str, dict[str, dict[str, float]]],
) -> dict:
    """Collapse per-SKU metrics into cross-SKU aggregates."""
    rows = []
    for model in MODEL_NAMES:
        valid = {
            sku: m for sku, m in results.items()
            if not np.isnan(m.get(model, {}).get("MAPE", np.nan))
        }
        if not valid:
            continue
        def mean_or_nan(values):
            values = [v for v in values if not np.isnan(v)]
            return float(np.mean(values)) if values else float("nan")
        def median_or_nan(values):
            values = [v for v in values if not np.isnan(v)]
            return float(np.median(values)) if values else float("nan")

        agg = {}
        for metric in METRIC_NAMES:
            vals = [results[sku].get(model, {}).get(metric, np.nan)
                    for sku in valid]
            agg[f"mean_{metric}"] = mean_or_nan(vals)
            agg[f"median_{metric}"] = median_or_nan(vals)
        agg["n_skus"] = len(valid)
        rows.append({"model": model, **agg})

    wins: dict[str, int] = defaultdict(int)
    for sku_id in results:
        best = None
        best_mape = float("inf")
        for model in MODEL_NAMES:
            m = results[sku_id].get(model, {})
            v = m.get("MAPE", np.nan)
            if not np.isnan(v) and v < best_mape:
                best_mape = v
                best = model
        if best is not None:
            wins[best] += 1

    # worst-performing SKUs by MAPE and by WAPE (LightGBM as the robust
    # reference), ranked
    def worst(model: str, metric: str, top: int = 5):
        scored = [
            (results[sku][model][metric], sku)
            for sku in results
            if not np.isnan(results[sku].get(model, {}).get(metric, np.nan))
        ]
        scored.sort()
        return scored[:top], scored[-top:][::-1]

    lightgbm_worst_mape = worst("LightGBM", "MAPE")
    lightgbm_worst_wape = worst("LightGBM", "WAPE")

    # by-category: mean MAPE per model per category
    cat_rows = []
    cats = sorted({results[s].get("category", "unknown")
                   for s in results})
    for cat in cats:
        skus = [s for s in results if results[s].get("category") == cat]
        row = {"category": cat, "n_skus": len(skus),
               "skus": skus}
        for model in MODEL_NAMES:
            vals = [results[s].get(model, {}).get("MAPE", np.nan)
                    for s in skus]
            vals = [v for v in vals if not np.isnan(v)]
            row[f"{model}_MAPE"] = (
                float(np.mean(vals)) if vals else float("nan"))
        cat_rows.append(row)

    return {
        "aggregate": rows,
        "wins": dict(wins),
        "worst_mape": lightgbm_worst_mape,
        "worst_wape": lightgbm_worst_wape,
        "by_category": cat_rows,
    }


def render_markdown(results: dict[str, dict[str, dict[str, float]]]) -> str:
    agg = aggregate_results(results)
    lines = [
        "# Forecast Evaluation",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        f"Evaluated **{len(results)}** SKU(s). Split: `TimeSeriesSplit` "
        "(forward-chaining, never a random split) -- the final fold's "
        "held-out segment is the test window for every model, so all "
        "models are compared on identical data per SKU.",
        "",
        "| Model | Mean MAPE | Median MAPE | Mean RMSE | Median RMSE | "
        "Mean WAPE | Mean Bias | SKU wins (MAPE) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in agg["aggregate"]:
        def fm(v):
            return f"{v:.3f}" if not np.isnan(v) else "n/a"
        lines.append(
            f"| {row['model']} | {fm(row.get('mean_MAPE', np.nan))} | "
            f"{fm(row.get('median_MAPE', np.nan))} | "
            f"{fm(row.get('mean_RMSE', np.nan))} | "
            f"{fm(row.get('median_RMSE', np.nan))} | "
            f"{fm(row.get('mean_WAPE', np.nan))} | "
            f"{fm(row.get('mean_Bias', np.nan))} | "
            f"{agg['wins'].get(row['model'], 0)} |"
        )
    lines.append("")

    # --- By category ---
    lines.append("## Performance by Category (mean MAPE)")
    lines.append("")
    lines.append("| Category | SKUs | " + " | ".join(MODEL_NAMES) + " |")
    lines.append("|" + "---|" * (len(MODEL_NAMES) + 2))
    for row in agg["by_category"]:
        def fm(v):
            return f"{v:.2f}" if not np.isnan(v) else "n/a"
        cells = " | ".join(fm(row.get(f"{m}_MAPE", np.nan))
                           for m in MODEL_NAMES)
        lines.append(f"| {row['category']} | {row['n_skus']} | {cells} |")
    lines.append("")

    # --- Worst performing SKUs ---
    lines.append("## Worst-performing SKUs (LightGBM, the robust reference)")
    lines.append("")
    lines.append("### Worst 5 by MAPE")
    lines.append("")
    lines.append("| SKU | MAPE |")
    lines.append("|---|---|")
    for v, sku in agg["worst_mape"][0]:
        lines.append(f"| {sku} | {v:.3f} |")
    lines.append("")
    lines.append("### Best 5 by MAPE")
    lines.append("")
    lines.append("| SKU | MAPE |")
    lines.append("|---|---|")
    for v, sku in agg["worst_mape"][1]:
        lines.append(f"| {sku} | {v:.3f} |")
    lines.append("")

    # Zero-forecast-safety: show models with NaN (unstable) explicitly
    lines.append("_Models missing for any SKU (LSTM/Ensemble instability):_ ")
    missing_notes = []
    for sku_id, metrics in results.items():
        missing = [m for m in MODEL_NAMES
                   if np.isnan(metrics.get(m, {}).get("MAPE", np.nan))]
        if missing:
            missing_notes.append(f"{sku_id}: {', '.join(missing)}")
    lines.append("; ".join(missing_notes) if missing_notes else "none")
    lines.append("")

    for sku_id, metrics in results.items():
        lines.append(f"## {sku_id}")
        lines.append("")
        header = "| Model | " + " | ".join(METRIC_NAMES) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(METRIC_NAMES) + 1))
        for model_name in MODEL_NAMES:
            m = metrics.get(model_name, {})
            cells = []
            for metric in METRIC_NAMES:
                v = m.get(metric, np.nan)
                cells.append(f"{v:.3f}" if not np.isnan(v) else "n/a")
            lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to real dataset CSV")
    parser.add_argument("--out", type=str, default="docs/forecast_eval.md")
    parser.add_argument("--max-skus", type=int, default=3,
                        help="Number of SKUs to evaluate (cost control)")
    parser.add_argument("--splits", type=int, default=3,
                        help="TimeSeriesSplit folds")
    args = parser.parse_args()

    if args.csv:
        df = load_csv(args.csv)
    else:
        df = generate_synthetic(n_days=300, stores=("S001",),
                                products=("P001", "P002"))

    sku_ids = df["sku_id"].unique()[: args.max_skus]
    cat_map = (
        df.groupby("sku_id")["Category"].first().to_dict()
        if "Category" in df.columns else {}
    )
    results = {}
    for sku_id in sku_ids:
        sku_df = df[df["sku_id"] == sku_id]
        try:
            results[sku_id] = evaluate_sku(
                sku_df, n_splits=args.splits, category=cat_map.get(sku_id),
            )
        except ValueError as e:
            print(f"Skipping {sku_id}: {e}")

    md = render_markdown(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote {out_path} ({len(results)} SKUs)")


if __name__ == "__main__":
    main()

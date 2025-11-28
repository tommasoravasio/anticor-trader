from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Minimal helper to persist backtest artifacts & summary.
# Call this at end of each dataset run.

def _safe_float(x: Any) -> Any:
    try:
        if isinstance(x, (int, float)):
            return float(x)
    except Exception:
        pass
    return x

COMMON_METRIC_NAMES = [
    "cumulative_return",
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "volatility",
    "avg_trade_return",
    "win_rate",
    "total_trade_pnl",
]


def collect_metrics(metrics_obj: Optional[Any]) -> Dict[str, Any]:
    """Normalize metrics input (dict / DataFrame / Series / object with to_dict)."""
    if metrics_obj is None:
        return {}
    data: Dict[str, Any] = {}
    try:
        if hasattr(metrics_obj, "to_dict"):
            raw = metrics_obj.to_dict()
        elif isinstance(metrics_obj, dict):
            raw = metrics_obj
        else:
            raw = {}
        for k, v in raw.items():
            data[str(k)] = _safe_float(v)
    except Exception as e:
        data["metrics_error"] = str(e)
    return data


def infer_additional_metrics(namespace: Dict[str, Any]) -> Dict[str, Any]:
    """Pick up loose metric variables in caller's namespace."""
    out: Dict[str, Any] = {}
    for name in COMMON_METRIC_NAMES:
        if name in namespace:
            out[name] = _safe_float(namespace[name])
    return out


def save_backtest_results(
    equity_df: Optional[pd.DataFrame],
    trades_df: Optional[pd.DataFrame],
    metrics_obj: Optional[Any],
    dataset_id: str,
    extra_namespace: Optional[Dict[str, Any]] = None,
    results_dir: str = "results/backtests",
) -> Dict[str, Any]:
    """
    Persist backtest artifacts & summary for later aggregation.

    Parameters
    ----------
    equity_df : DataFrame of equity curve (index: datetime or step).
    trades_df : DataFrame of executed trades (must have at least one row to count trades).
    metrics_obj : Dict / DataFrame / object with to_dict holding performance metrics.
    dataset_id : Identifier for dataset; used in filenames.
    extra_namespace : Optionally pass globals() to auto-grab standalone metric variables.
    results_dir : Output directory (created if missing).

    Returns
    -------
    summary : Dict of collected metrics & artifact file paths.
    """
    dataset_id = dataset_id or "dataset_unknown"
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"dataset_id": dataset_id}

    # Equity curve
    if equity_df is not None:
        try:
            equity_path = out_dir / f"{dataset_id}_equity.csv"
            equity_df.to_csv(equity_path)
            summary["equity_curve_file"] = str(equity_path)
            # Basic equity stats if possible
            if "equity" in equity_df.columns:
                eq = equity_df["equity"].astype(float)
                summary["final_equity"] = _safe_float(eq.iloc[-1])
                summary["peak_equity"] = _safe_float(eq.max())
        except Exception as e:
            summary["equity_curve_error"] = str(e)

    # Trades
    if trades_df is not None:
        try:
            trades_path = out_dir / f"{dataset_id}_trades.csv"
            trades_df.to_csv(trades_path)
            summary["trades_file"] = str(trades_path)
            summary["num_trades"] = int(len(trades_df))
            if "PnL" in trades_df.columns:
                pnl = trades_df["PnL"].astype(float)
                summary["total_trade_pnl"] = _safe_float(pnl.sum())
                summary["win_rate"] = _safe_float((pnl > 0).mean())
        except Exception as e:
            summary["trades_error"] = str(e)

    # Structured metrics object
    summary.update(collect_metrics(metrics_obj))

    # Loose metrics from namespace
    if extra_namespace:
        summary.update(infer_additional_metrics(extra_namespace))

    # Persist JSON summary
    summary_path = out_dir / f"{dataset_id}_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    summary["summary_file"] = str(summary_path)

    # Append to master JSONL
    master_path = out_dir / "all_summaries.jsonl"
    with master_path.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")

    return summary


__all__ = ["save_backtest_results"]

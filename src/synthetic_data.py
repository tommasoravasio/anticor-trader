"""
Synthetic ANTICOR experiments on semi-random markets (synthetic data).

This script reuses the ANTICOR(w) and ANTI¹ logic from the main notebook and
applies it to synthetic datasets with different behaviors:
- clear trending winner
- mean-reverting noise
- rotating leadership (no single winner)
- choppy blocks with frequent rank flips

Extras added here:
- parameter sweeps across seeds and windows
- CSV/JSON exports of metrics
- quick equity plots for a reference run
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import os

import numpy as np
import pandas as pd
import math

# Matplotlib needs a writable cache directory in sandboxed environments.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core strategies (copied from the optimized notebook, condensed for reuse)
# ---------------------------------------------------------------------------
def run_anticor_strategy(
    price_relatives: pd.DataFrame,
    w: int,
    permno_to_ticker: Dict[str, str] | None = None,
    debug_days: int = 0,
    returns_only: bool = False,
    force_start_at_index: int | None = None,
    dtype=np.float32,
) -> pd.DataFrame:
    """
    Vectorized ANTICOR(w) backtest.

    Parameters
    ----------
    price_relatives : DataFrame of (1 + returns), index is dates, columns are assets.
    w : window length.
    permno_to_ticker : optional mapping to readable labels.
    debug_days : print the first N days of trading context.
    returns_only : skip weight history to speed up ANTI¹ layer.
    force_start_at_index : align start date across experts.
    dtype : internal float type.
    """
    permno_to_ticker = permno_to_ticker or {c: c for c in price_relatives.columns}

    dates = price_relatives.index
    assets = price_relatives.columns
    m = price_relatives.shape[1]
    n = len(price_relatives)

    pr = price_relatives.to_numpy(dtype=dtype, copy=False)
    log_relatives = np.log(pr).astype(dtype, copy=False)

    t_start = 2 * w
    if force_start_at_index is not None:
        t_start = max(t_start, int(force_start_at_index))
    if t_start >= n:
        raise ValueError("Not enough observations for the requested window.")

    out_len = n - t_start
    daily_returns_history = np.empty(out_len, dtype=dtype)
    dates_history = np.empty(out_len, dtype=object)
    store_weights = not returns_only
    if store_weights:
        weights_history = np.empty((out_len, m), dtype=dtype)

    b_t = np.full(m, 1.0 / m, dtype=dtype)

    for k, t in enumerate(range(t_start, n)):
        is_debug_day = k < debug_days

        day_ret = (b_t * pr[t]).sum() - 1.0
        daily_returns_history[k] = day_ret
        dates_history[k] = dates[t]
        if store_weights:
            weights_history[k, :] = b_t

        b_t_reb = b_t * pr[t]
        denom = b_t_reb.sum()
        if denom <= 0:
            b_t_reb = np.full(m, 1.0 / m, dtype=dtype)
        else:
            b_t_reb = b_t_reb / denom

        X1 = log_relatives[t - 2 * w + 1 : t - w + 1, :]
        X2 = log_relatives[t - w + 1 : t + 1, :]

        mu2 = X2.mean(axis=0)
        X1_mean = X1.mean(axis=0)
        X2_mean = X2.mean(axis=0)
        X1_std = X1.std(axis=0, ddof=1)
        X2_std = X2.std(axis=0, ddof=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            X1z = (X1 - X1_mean) / X1_std
            X2z = (X2 - X2_mean) / X2_std
        X1z = np.where(np.isfinite(X1z), X1z, 0.0)
        X2z = np.where(np.isfinite(X2z), X2z, 0.0)

        m_cor = (X1z.T @ X2z) / max(w - 1, 1)

        mu_cmp = mu2[:, None] > mu2[None, :]
        pos_corr = m_cor > 0
        offdiag = ~np.eye(m, dtype=bool)

        diag = np.diag(m_cor)
        extra = -np.minimum(diag, 0.0)
        claims = m_cor + extra[:, None] + extra[None, :]
        mask = offdiag & mu_cmp & pos_corr
        claims = np.where(mask, claims, 0.0)

        total_out = claims.sum(axis=1)
        has_out = total_out > 0
        P = np.zeros_like(claims, dtype=dtype)
        P[has_out, :] = claims[has_out, :] / total_out[has_out, None]

        b_next = np.zeros_like(b_t_reb)
        b_next += b_t_reb @ P
        b_next += b_t_reb * (~has_out)

        s = b_next.sum()
        if s <= 0:
            b_t = np.full(m, 1.0 / m, dtype=dtype)
        else:
            b_t = b_next / s

        if is_debug_day:
            top_idx = np.argsort(-b_t_reb)[:5]
            tickers = [permno_to_ticker.get(assets[i], str(assets[i])) for i in top_idx]
            print(f"\n===== TRADING DAY: {dates[t]} (t={t}) =====")
            print("Top weights (after rebalance):")
            for i, wt in zip(tickers, b_t_reb[top_idx]):
                print(f"  - {i:<8}: {wt:.2%}")

    results_df = pd.DataFrame(daily_returns_history, index=pd.Index(dates_history), columns=["daily_profit"])

    if store_weights:
        col_names = [f"weight_{permno_to_ticker.get(a, a)}" for a in assets]
        weights_df = pd.DataFrame(weights_history, index=results_df.index, columns=col_names)
        return pd.concat([results_df, weights_df], axis=1)
    else:
        return results_df


def run_anti1_strategy(price_relatives: pd.DataFrame, max_W: int, permno_to_ticker: Dict[str, str]) -> pd.DataFrame:
    """
    Smoothed ANTI¹ = average of ANTICOR(w) experts for w in [2, max_W].
    All experts start on the same aligned date.
    """
    start_idx = 2 * max_W
    if start_idx >= len(price_relatives.index):
        raise ValueError("Not enough observations for ANTI¹ with this W_max.")
    start_date = price_relatives.index[start_idx]

    sum_weights = None
    expert_count = 0

    for w in range(2, max_W + 1):
        temp_results = run_anticor_strategy(
            price_relatives, w, permno_to_ticker, debug_days=0, returns_only=False, force_start_at_index=start_idx
        )
        df_weights = temp_results.filter(like="weight_")
        if sum_weights is None:
            sum_weights = df_weights.copy()
        else:
            sum_weights = sum_weights.add(df_weights, fill_value=0.0)
        expert_count += 1

    mean_weights_df = sum_weights / float(expert_count)

    price_relatives_renamed = price_relatives.rename(columns=permno_to_ticker)
    daily_returns_assets = price_relatives_renamed.loc[start_date:] - 1
    daily_returns_assets.columns = [f"weight_{col}" for col in daily_returns_assets.columns]

    anti1_daily_returns = (mean_weights_df * daily_returns_assets).sum(axis=1)

    results_df = pd.DataFrame(anti1_daily_returns, columns=["daily_profit"])
    results_df["profit_cumule"] = (1 + results_df["daily_profit"]).cumprod()
    results_df = results_df.join(mean_weights_df)
    return results_df


def run_anti2_strategy(price_relatives: pd.DataFrame, max_W: int, permno_to_ticker: Dict[str, str]) -> pd.DataFrame:
    """
    Composite ANTI² (BAH(ANTICOR(ANTICOR))).
    Layer 1: ANTICOR(w) experts on assets (returns_only).
    Layer 2: ANTI¹ on the experts' return streams.
    """
    start_idx = 2 * max_W
    if start_idx >= len(price_relatives.index):
        raise ValueError("Not enough observations for ANTI² with this W_max.")

    expert_daily_returns: Dict[str, pd.Series] = {}
    for w in range(2, max_W + 1):
        temp_results = run_anticor_strategy(
            price_relatives,
            w,
            permno_to_ticker,
            debug_days=0,
            returns_only=True,
            force_start_at_index=start_idx,
        )
        expert_daily_returns[f"w_{w}"] = temp_results["daily_profit"]

    anti1_returns_df = pd.DataFrame(expert_daily_returns)

    anti1_results_on_experts = run_anti1_strategy(
        price_relatives=anti1_returns_df + 1.0,  # convert back to relatives
        max_W=max_W,
        permno_to_ticker={c: c for c in anti1_returns_df.columns},
    )
    return anti1_results_on_experts


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------
def _clip_to_price_relatives(raw_returns: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw_returns, -0.95, None)
    return 1.0 + clipped


def _make_index(num_days: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=num_days, freq="B")


def generate_trending_winner(num_assets: int, num_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drifts = np.linspace(0.0025, 0.0005, num_assets)  # clear ranking
    noise = rng.normal(0, 0.012, size=(num_days, num_assets))
    returns = noise + drifts
    pr = _clip_to_price_relatives(returns)
    cols = [f"Trend_{i+1}" for i in range(num_assets)]
    return pd.DataFrame(pr, index=_make_index(num_days), columns=cols)


def generate_mean_reversion(num_assets: int, num_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    phi = -0.4
    noise_sd = 0.01
    returns = np.zeros((num_days, num_assets))
    for t in range(1, num_days):
        shock = rng.normal(0, noise_sd, size=num_assets)
        returns[t] = phi * returns[t - 1] + shock
    pr = _clip_to_price_relatives(returns)
    cols = [f"MeanRev_{i+1}" for i in range(num_assets)]
    return pd.DataFrame(pr, index=_make_index(num_days), columns=cols)


def generate_rotating_leaders(num_assets: int, num_days: int, seed: int, block: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = np.zeros((num_days, num_assets))
    leader = 0
    for start in range(0, num_days, block):
        leader = rng.integers(0, num_assets)
        leader_drift = rng.uniform(0.002, 0.004)
        lag_drift = -rng.uniform(0.0005, 0.0015)
        end = min(start + block, num_days)
        for t in range(start, end):
            shocks = rng.normal(0, 0.012, size=num_assets)
            drift = np.full(num_assets, lag_drift)
            drift[leader] = leader_drift
            returns[t] = drift + shocks
    pr = _clip_to_price_relatives(returns)
    cols = [f"Rotate_{i+1}" for i in range(num_assets)]
    return pd.DataFrame(pr, index=_make_index(num_days), columns=cols)


def generate_choppy_blocks(num_assets: int, num_days: int, seed: int, block: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = np.zeros((num_days, num_assets))
    for start in range(0, num_days, block):
        block_trend = rng.normal(0, 0.001, size=num_assets)
        end = min(start + block, num_days)
        shocks = rng.normal(0, 0.015, size=(end - start, num_assets))
        returns[start:end] = block_trend + shocks
    pr = _clip_to_price_relatives(returns)
    cols = [f"Choppy_{i+1}" for i in range(num_assets)]
    return pd.DataFrame(pr, index=_make_index(num_days), columns=cols)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())


def summarize_performance(daily_returns: pd.Series) -> Dict[str, float]:
    gross = float((1 + daily_returns).prod())
    ann_factor = 252 / len(daily_returns)
    ann_return = float((gross ** ann_factor) - 1.0)
    vol = float(daily_returns.std(ddof=1) * np.sqrt(252))
    sharpe = float(ann_return / vol) if vol > 0 else 0.0
    equity = (1 + daily_returns).cumprod()
    mdd = max_drawdown(equity)
    return {
        "cumulative_return": gross,
        "annualized_return": ann_return,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


@dataclass
class Scenario:
    name: str
    generator: Callable[[int], pd.DataFrame]
    description: str


def build_scenarios(num_assets: int, num_days: int) -> List[Scenario]:
    """Central place to register scenarios so sweeps stay consistent."""
    return [
        Scenario(
            "trend_dominant",
            lambda seed: generate_trending_winner(num_assets, num_days, seed),
            "One clear trending winner with laggards (should hurt ANTICOR).",
        ),
        Scenario(
            "mean_reversion",
            lambda seed: generate_mean_reversion(num_assets, num_days, seed),
            "Assets bounce around a mean; leadership keeps flipping on shocks.",
        ),
        Scenario(
            "rotating_leaders",
            lambda seed: generate_rotating_leaders(num_assets, num_days, seed),
            "Winner rotates every few weeks; no permanent champion.",
        ),
        Scenario(
            "choppy_blocks",
            lambda seed: generate_choppy_blocks(num_assets, num_days, seed),
            "Block-level drifts with noise create frequent rank reshuffling.",
        ),
    ]


@dataclass
class ExperimentRun:
    scenario: str
    seed: int
    w: int
    W_max: int
    metrics_anticor: Dict[str, float]
    metrics_anti1: Dict[str, float]
    metrics_anti2: Dict[str, float]
    rank_churn: float
    equity_anticor: pd.Series
    equity_anti1: pd.Series
    equity_anti2: pd.Series

    def to_record(self) -> Dict[str, float]:
        rec = {
            "scenario": self.scenario,
            "seed": self.seed,
            "w": self.w,
            "W_max": self.W_max,
            "rank_churn": self.rank_churn,
        }
        rec.update({f"anticor_{k}": v for k, v in self.metrics_anticor.items()})
        rec.update({f"anti1_{k}": v for k, v in self.metrics_anti1.items()})
        rec.update({f"anti2_{k}": v for k, v in self.metrics_anti2.items()})
        return rec


def run_single_scenario(
    scen: Scenario,
    seed: int,
    w: int,
    W_max: int,
    save_data_dir: Path | None = None,
) -> ExperimentRun:
    df = scen.generator(seed)
    mapping = {col: col for col in df.columns}

    anticor_df = run_anticor_strategy(df, w=w, permno_to_ticker=mapping, debug_days=0)
    anti1_df = run_anti1_strategy(df, max_W=W_max, permno_to_ticker=mapping)
    anti2_df = run_anti2_strategy(df, max_W=W_max, permno_to_ticker=mapping)

    anticor_metrics = summarize_performance(anticor_df["daily_profit"])
    anti1_metrics = summarize_performance(anti1_df["daily_profit"])
    anti2_metrics = summarize_performance(anti2_df["daily_profit"])
    rotation_score = float(df.pct_change().rank(axis=1).diff().abs().mean().mean())

    equity_anticor = (1 + anticor_df["daily_profit"]).cumprod()
    equity_anti1 = (1 + anti1_df["daily_profit"]).cumprod()
    equity_anti2 = (1 + anti2_df["daily_profit"]).cumprod()

    if save_data_dir is not None:
        save_data_dir.mkdir(parents=True, exist_ok=True)
        data_path = save_data_dir / f"{scen.name}_seed{seed}_w{w}_data.csv"
        df.to_csv(data_path)
        print(f"Saved synthetic data for {scen.name} (seed={seed}, w={w}) to {data_path}")

    return ExperimentRun(
        scenario=scen.name,
        seed=seed,
        w=w,
        W_max=W_max,
        metrics_anticor=anticor_metrics,
        metrics_anti1=anti1_metrics,
        metrics_anti2=anti2_metrics,
        rank_churn=rotation_score,
        equity_anticor=equity_anticor,
        equity_anti1=equity_anti1,
        equity_anti2=equity_anti2,
    )


def print_human_readable(run: ExperimentRun) -> None:
    print(f"\n================= Scenario: {run.scenario} (seed={run.seed}, w={run.w}) =================")
    print("ANTICOR metrics:")
    for k, v in run.metrics_anticor.items():
        print(f"  {k:<18}: {v:.4f}")
    print("ANTI¹ metrics:")
    for k, v in run.metrics_anti1.items():
        print(f"  {k:<18}: {v:.4f}")
    print("ANTI² metrics:")
    for k, v in run.metrics_anti2.items():
        print(f"  {k:<18}: {v:.4f}")
    print(f"Mean rank-change magnitude (rough churn proxy): {run.rank_churn:.4f}")


def sweep_scenarios(
    scenarios: List[Scenario],
    seeds: List[int],
    ws: List[int],
    W_max: int,
    out_dir: Path,
) -> Tuple[List[ExperimentRun], List[Tuple[str, Dict[str, pd.Series]]]]:
    """
    Run a small sweep across seeds and window lengths.
    Returns list of ExperimentRun and a few equity curves for plotting.
    """
    runs: List[ExperimentRun] = []
    sample_equities: List[Tuple[str, Dict[str, pd.Series]]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        for w in ws:
            for scen in scenarios:
                run = run_single_scenario(scen, seed=seed, w=w, W_max=W_max)
                runs.append(run)
                # Keep a small subset for plots (first seed and first w).
                if seed == seeds[0] and w == ws[0]:
                    sample_equities.append(
                        (
                            scen.name,
                            {
                                "ANTICOR": run.equity_anticor,
                                "ANTI1": run.equity_anti1,
                                "ANTI2": run.equity_anti2,
                            },
                        )
                    )
    return runs, sample_equities


def save_results_table(runs: List[ExperimentRun], out_dir: Path) -> None:
    df = pd.DataFrame([r.to_record() for r in runs])
    csv_path = out_dir / "synthetic_metrics.csv"
    json_path = out_dir / "synthetic_metrics.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\nSaved sweep results to: {csv_path} and {json_path}")
    grouped = df.groupby("scenario").mean(numeric_only=True)
    summary_cols = [
        "anticor_sharpe",
        "anticor_max_drawdown",
        "anti1_sharpe",
        "anti1_max_drawdown",
        "anti2_sharpe",
        "anti2_max_drawdown",
        "rank_churn",
    ]
    print("\nScenario summary (Sharpe, drawdown, churn):")
    print(grouped[summary_cols])


def plot_sample_equities(equity_sets: List[Tuple[str, Dict[str, pd.Series]]], out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for scen_name, curves in equity_sets:
        plt.figure(figsize=(8, 4))
        for label, series in curves.items():
            plt.plot(series.index, series.values, label=label)
        plt.title(f"Equity curves - {scen_name}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative wealth")
        plt.legend()
        plt.tight_layout()
        fname = plot_dir / f"{scen_name}_equity.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved plot: {fname}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_scenarios(num_assets: int = 8, num_days: int = 500, w: int = 30, W_max: int = 30) -> None:
    base_seed = 42
    scenarios = build_scenarios(num_assets=num_assets, num_days=num_days)

    for idx, scen in enumerate(scenarios):
        run = run_single_scenario(
            scen,
            seed=base_seed + idx,
            w=w,
            W_max=W_max,
            save_data_dir=Path("results/synthetic/data"),
        )
        print_human_readable(run)


def main() -> None:
    # Default quick run
    run_scenarios()

    # Richer sweep (small and fast) with exports + plots.
    seeds = [101, 202, 303]
    ws = [30]  # match main notebook window choice
    scenarios = build_scenarios(num_assets=10, num_days=600)
    out_dir = Path("results/synthetic")
    runs, sample_equities = sweep_scenarios(
        scenarios=scenarios,
        seeds=seeds,
        ws=ws,
        W_max=30,
        out_dir=out_dir,
    )
    save_results_table(runs, out_dir)
    plot_sample_equities(sample_equities, out_dir)


if __name__ == "__main__":
    main()

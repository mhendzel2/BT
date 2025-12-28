import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    lookback: int
    holding: int
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    trades: int
    equity_curve: pd.Series
    period_returns: pd.Series


def _prep_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame = frame.dropna(axis=1, how="all")
    frame = frame.ffill().dropna()
    return frame


def _equity_and_metrics(period_returns: pd.Series, holding: int) -> Dict[str, float]:
    equity = (1 + period_returns).cumprod()
    if equity.empty:
        return {
            "equity": equity,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
        }
    total_days = holding * len(period_returns)
    years = total_days / 252
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    periods_per_year = 252 / holding if holding else 0
    std = period_returns.std()
    sharpe = period_returns.mean() / std * math.sqrt(periods_per_year) if std and not math.isnan(std) else 0.0
    max_dd = ((equity / equity.cummax()) - 1).min()
    win_rate = (period_returns > 0).mean() if len(period_returns) else 0.0
    return {"equity": equity, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate}


def run_momentum_backtest(
    prices: pd.DataFrame,
    lookback: int,
    holding: int,
    top_n: int = 5,
    bottom_n: int = 0,
    regime_filter: Optional[pd.Series] = None,
) -> BacktestResult:
    closes = _prep_prices(prices)
    index = closes.index
    period_returns: List[float] = []
    period_dates: List[pd.Timestamp] = []
    trades = 0

    step = max(1, holding)
    for i in range(lookback, len(index) - step, step):
        date = index[i]
        if regime_filter is not None and date in regime_filter.index and not bool(regime_filter.loc[date]):
            period_returns.append(0.0)
            period_dates.append(index[i + step])
            continue

        window = closes.iloc[i - lookback : i]
        momentum = window.iloc[-1] / window.iloc[0] - 1
        momentum = momentum.dropna()
        if momentum.empty:
            continue

        winners = momentum.nlargest(top_n).index if top_n else []
        losers = momentum.nsmallest(bottom_n).index if bottom_n else []
        total_positions = len(winners) + len(losers)
        if total_positions == 0:
            continue

        future = closes.iloc[i : i + step]
        future_return = future.iloc[-1] / future.iloc[0] - 1

        weights = {}
        if len(winners) > 0:
            weight = 1 / total_positions
            for w in winners:
                weights[w] = weight
        if len(losers) > 0:
            weight = -1 / total_positions
            for l in losers:
                weights[l] = weight

        aligned = future_return.reindex(weights.keys()).fillna(0)
        period_ret = (aligned * pd.Series(weights)).sum()
        period_returns.append(period_ret)
        period_dates.append(index[i + step])
        trades += total_positions

    period_returns_series = pd.Series(period_returns, index=period_dates)
    metrics = _equity_and_metrics(period_returns_series, holding)

    return BacktestResult(
        lookback=lookback,
        holding=holding,
        cagr=float(metrics["cagr"]),
        sharpe=float(metrics["sharpe"]),
        max_drawdown=float(metrics["max_dd"]),
        win_rate=float(metrics["win_rate"]),
        trades=trades,
        equity_curve=metrics["equity"],
        period_returns=period_returns_series,
    )


def optimize_grid(
    prices: pd.DataFrame,
    lookbacks: Sequence[int],
    holdings: Sequence[int],
    top_n: int,
    bottom_n: int,
    regime_filter: Optional[pd.Series] = None,
) -> pd.DataFrame:
    records = []
    for lb in lookbacks:
        for hd in holdings:
            result = run_momentum_backtest(
                prices=prices,
                lookback=lb,
                holding=hd,
                top_n=top_n,
                bottom_n=bottom_n,
                regime_filter=regime_filter,
            )
            records.append(
                {
                    "lookback": lb,
                    "holding": hd,
                    "cagr": result.cagr,
                    "sharpe": result.sharpe,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "trades": result.trades,
                    "equity_curve": result.equity_curve,
                    "period_returns": result.period_returns,
                }
            )
    return pd.DataFrame(records)


def best_parameter_set(results: pd.DataFrame, metric: str = "cagr") -> Optional[Dict]:
    if results.empty:
        return None
    ordered = results.sort_values(metric, ascending=False)
    return ordered.iloc[0].to_dict()

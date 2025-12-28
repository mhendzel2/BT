import datetime as dt
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from backtester import best_parameter_set, optimize_grid
from data_loader import DEFAULT_TICKERS, fetch_market_data
from regime import classify_regime, regime_mask


st.set_page_config(page_title="Momentum Optimizer (IBKR/YFinance)", layout="wide")
st.title("Momentum Optimizer: Trending vs Consolidating")
st.caption("Optimize lookback and holding periods for different market regimes. Data from IBKR TWS or Yahoo Finance.")


@st.cache_data(show_spinner=False)
def load_data(tickers: List[str], start: dt.date, end: dt.date, source: str, **kwargs):
    return fetch_market_data(tickers, start, end, source=source, **kwargs)


def make_heatmap(results: pd.DataFrame, title: str):
    if results.empty:
        return None
    pivot = results.pivot(index="lookback", columns="holding", values="cagr")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="GnBu",
        origin="lower",
        labels={"color": "CAGR", "x": "Holding", "y": "Lookback"},
        title=title,
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


def plot_equity_curve(curve: pd.Series, title: str):
    if curve is None or curve.empty:
        return None
    fig = px.line(curve, title=title, labels={"value": "Equity", "index": "Date"})
    return fig


def metrics_columns(result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{result['cagr']:.2%}")
    c2.metric("Sharpe", f"{result['sharpe']:.2f}")
    c3.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")
    c4.metric("Win Rate", f"{result['win_rate']:.2%}")


with st.sidebar:
    st.header("Parameters")
    data_source = st.selectbox("Data Source", ["yfinance", "ibkr"], index=0)
    ib_host = st.text_input("IB host", "127.0.0.1") if data_source == "ibkr" else None
    ib_port = (
        st.number_input("IB port", min_value=1000, max_value=10000, value=7497, step=1)
        if data_source == "ibkr"
        else None
    )
    ib_client = (
        st.number_input("IB client id", min_value=1, max_value=100000, value=101, step=1)
        if data_source == "ibkr"
        else None
    )
    tickers = st.multiselect("Universe (tickers)", options=DEFAULT_TICKERS, default=DEFAULT_TICKERS[:10])
    start_date = st.date_input("Start date", dt.date(2018, 1, 1))
    end_date = st.date_input("End date", dt.date.today())
    top_n = st.slider("Top N longs", 1, 10, 5)
    bottom_n = st.slider("Bottom N shorts", 0, 10, 0)
    lookback_range = st.slider("Lookback window (days)", 5, 120, (20, 60), step=5)
    holding_range = st.slider("Holding period (days)", 5, 90, (10, 40), step=5)
    grid_step = st.slider("Grid step", 5, 20, 10, step=5)
    chop_threshold = st.slider("Choppiness threshold (lower = more trending)", 40.0, 70.0, 55.0, step=1.0)
    regime_period = st.slider("Regime lookback for choppiness", 10, 30, 14)
    run_btn = st.button("Run optimization")


if run_btn:
    with st.spinner("Downloading data..."):
        kwargs = {}
        if data_source == "ibkr":
            kwargs.update({"host": ib_host, "port": int(ib_port), "client_id": int(ib_client)})
        data = load_data(tickers, start_date, end_date, data_source, **kwargs)

    available = [t for t in tickers if t in data["close"].columns]
    if not available:
        st.error("No price data returned for selected tickers.")
        st.stop()
    closes = data["close"][available]
    benchmark_symbol = data.get("benchmark", "SPY")
    bench_close = data["close"].get(benchmark_symbol, None)
    bench_high = data["high"].get(benchmark_symbol, None)
    bench_low = data["low"].get(benchmark_symbol, None)

    if bench_close is None or bench_high is None or bench_low is None:
        st.error("Benchmark data missing. Try again with yfinance or add SPY to the universe.")
        st.stop()

    regime_df = classify_regime(
        high=bench_high,
        low=bench_low,
        close=bench_close,
        period=regime_period,
        trend_threshold=chop_threshold,
    )

    st.subheader("Market Regime")
    regime_fig = px.line(
        regime_df["chop"],
        title="Choppiness Index (lower = trending)",
        labels={"value": "Choppiness", "index": "Date"},
    )
    regime_fig.add_hrect(y0=0, y1=chop_threshold, line_width=0, fillcolor="green", opacity=0.1)
    st.plotly_chart(regime_fig, use_container_width=True)

    lookbacks = list(range(lookback_range[0], lookback_range[1] + 1, grid_step))
    holdings = list(range(holding_range[0], holding_range[1] + 1, grid_step))

    st.subheader("Optimization Results")
    col1, col2 = st.columns(2)

    with st.spinner("Optimizing for trending periods..."):
        trending_results = optimize_grid(
            closes,
            lookbacks,
            holdings,
            top_n=top_n,
            bottom_n=bottom_n,
            regime_filter=regime_mask(regime_df, "trending"),
        )
    with st.spinner("Optimizing for consolidating periods..."):
        consolidating_results = optimize_grid(
            closes,
            lookbacks,
            holdings,
            top_n=top_n,
            bottom_n=bottom_n,
            regime_filter=regime_mask(regime_df, "consolidating"),
        )

    trend_heatmap = make_heatmap(trending_results, "Trending regime CAGR by lookback/holding")
    chop_heatmap = make_heatmap(consolidating_results, "Consolidating regime CAGR by lookback/holding")
    if trend_heatmap:
        col1.plotly_chart(trend_heatmap, use_container_width=True)
    if chop_heatmap:
        col2.plotly_chart(chop_heatmap, use_container_width=True)

    best_trend = best_parameter_set(trending_results, metric="cagr")
    best_chop = best_parameter_set(consolidating_results, metric="cagr")

    st.subheader("Best parameter sets")
    b1, b2 = st.columns(2)
    if best_trend:
        b1.write(f"**Trending:** lookback {int(best_trend['lookback'])}d | holding {int(best_trend['holding'])}d")
        metrics_columns(best_trend)
    else:
        b1.info("No trades in trending regime for selected inputs.")

    if best_chop:
        b2.write(f"**Consolidating:** lookback {int(best_chop['lookback'])}d | holding {int(best_chop['holding'])}d")
        metrics_columns(best_chop)
    else:
        b2.info("No trades in consolidating regime for selected inputs.")

    st.subheader("Equity curves")
    eq_col1, eq_col2 = st.columns(2)
    if best_trend:
        eq_fig = plot_equity_curve(best_trend["equity_curve"], "Trending best equity curve")
        if eq_fig:
            eq_col1.plotly_chart(eq_fig, use_container_width=True)
    if best_chop:
        eq_fig2 = plot_equity_curve(best_chop["equity_curve"], "Consolidating best equity curve")
        if eq_fig2:
            eq_col2.plotly_chart(eq_fig2, use_container_width=True)

    st.subheader("Sample trades per period")
    sample = []
    if best_trend:
        sample.append(
            {
                "Regime": "Trending",
                "Lookback": int(best_trend["lookback"]),
                "Holding": int(best_trend["holding"]),
                "Trades": int(best_trend["trades"]),
                "CAGR": best_trend["cagr"],
            }
        )
    if best_chop:
        sample.append(
            {
                "Regime": "Consolidating",
                "Lookback": int(best_chop["lookback"]),
                "Holding": int(best_chop["holding"]),
                "Trades": int(best_chop["trades"]),
                "CAGR": best_chop["cagr"],
            }
        )
    if sample:
        st.dataframe(pd.DataFrame(sample))
else:
    st.info("Set parameters on the left and click Run optimization to explore holding vs lookback across regimes.")

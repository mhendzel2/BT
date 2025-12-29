import datetime as dt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


st.title("Outlier Follow-Through Analyzer")
st.caption(
    "Load outlier files, filter by date range, pull surrounding price action from Yahoo Finance, "
    "and inspect follow-through and correlations."
)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "ticker" not in df.columns:
        if "underlyingsymbol" in df.columns:
            df = df.rename(columns={"underlyingsymbol": "ticker"})
    return df


def _date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["date", "currdate", "asofdate", "trade_date", "event_date"]:
        if c in df.columns:
            return c
    return None


def _read_outliers(uploads: Iterable[Union[str, object]]) -> Tuple[pd.DataFrame, Optional[str]]:
    frames = []
    date_col = None
    for up in uploads:
        try:
            df = pd.read_csv(up)
        except Exception as exc:  # pragma: no cover - Streamlit surface
            name = up if isinstance(up, (str, Path)) else getattr(up, "name", "file")
            st.error(f"Could not read {name}: {exc}")
            continue
        df = _normalize(df)
        col = _date_col(df)
        if col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            date_col = date_col or col
        frames.append(df)
    if not frames:
        return pd.DataFrame(), date_col
    merged = pd.concat(frames, ignore_index=True)
    return merged, date_col


def _pick_magnitude(df: pd.DataFrame) -> Optional[str]:
    for c in ["z_score", "manip_score", "oidiff", "oi_change", "oi_change_atm"]:
        if c in df.columns:
            return c
    return None


def _fetch_prices(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(sorted(set(tickers)), start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Adj Close"].fillna(raw.get("Close"))
    else:
        name = tickers[0] if len(tickers) == 1 else "price"
        close = raw[["Adj Close"]].rename(columns={"Adj Close": name})
    close = close.dropna(how="all")
    return close


def _event_metrics(outliers: pd.DataFrame, date_col: str, prices: pd.DataFrame, magnitude_col: Optional[str]):
    records = []
    windows = []
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()
    for _, row in outliers.iterrows():
        ticker = row.get("ticker")
        event_date = row.get(date_col)
        if pd.isna(ticker) or pd.isna(event_date):
            continue
        ticker = str(ticker).upper()
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna().sort_index()
        if series.empty:
            continue
        anchor = series.asof(event_date)
        if pd.isna(anchor):
            continue
        window = series.loc[event_date - pd.Timedelta(days=7) : event_date + pd.Timedelta(days=7)]
        rel = (window / anchor) - 1
        for idx, val in rel.items():
            windows.append(
                {
                    "ticker": ticker,
                    "event_date": pd.to_datetime(event_date),
                    "t_offset": int((idx - pd.to_datetime(event_date)).days),
                    "rel_return": val,
                }
            )
        rec = {
            "ticker": ticker,
            "event_date": pd.to_datetime(event_date),
            "anchor_price": anchor,
        }
        for h in [1, 3, 5, 7]:
            fwd_price = series.asof(event_date + pd.Timedelta(days=h))
            rec[f"fwd_{h}d"] = (fwd_price / anchor - 1) if pd.notna(fwd_price) else np.nan
        if magnitude_col:
            rec["magnitude"] = row.get(magnitude_col)
        records.append(rec)
    return pd.DataFrame(records), pd.DataFrame(windows)


with st.sidebar:
    st.subheader("Inputs")
    outlier_uploads = st.file_uploader("Outlier CSVs (multiple allowed)", type=["csv"], accept_multiple_files=True)
    dir_path = st.text_input("Directory with outlier CSVs", value="", help="Optional: load all CSVs from this folder.")
    default_start = dt.date(2025, 11, 24)
    default_end = dt.date(2025, 12, 10)
    date_range = st.date_input("Outlier date window", (default_start, default_end))
    run_btn = st.button("Run follow-through")

if run_btn:
    file_inputs: List[Union[str, Path, object]] = list(outlier_uploads or [])
    if dir_path.strip():
        folder = Path(dir_path).expanduser()
        if folder.is_dir():
            file_inputs.extend(sorted(folder.glob("*.csv")))
            st.info(f"Loaded {len(list(folder.glob('*.csv')))} CSV files from {folder}")
        else:
            st.warning(f"Directory not found: {folder}")

    outliers, date_col = _read_outliers(file_inputs)
    if outliers.empty or date_col is None:
        st.error("Upload at least one outlier CSV with a date column.")
        st.stop()

    start_date, end_date = date_range if isinstance(date_range, tuple) else (default_start, default_end)
    mask = outliers[date_col].dt.date.between(start_date, end_date)
    scoped = outliers[mask].copy()
    scoped["ticker"] = scoped["ticker"].astype(str).str.upper()
    if scoped.empty:
        st.warning("No outliers in the selected window.")
        st.stop()

    magnitude_col = _pick_magnitude(scoped)
    st.write(f"Using outlier magnitude: `{magnitude_col or 'none found'}`")

    buffer_start = start_date - dt.timedelta(days=7)
    buffer_end = end_date + dt.timedelta(days=7)
    tickers = sorted(scoped["ticker"].dropna().unique().tolist())
    prices = _fetch_prices(tickers, buffer_start, buffer_end)

    if prices.empty:
        st.error("No price data returned from Yahoo Finance.")
        st.stop()

    metrics, windows = _event_metrics(scoped, date_col, prices, magnitude_col)

    st.subheader("Event-level returns")
    if metrics.empty:
        st.info("No events with matching prices.")
        st.stop()
    st.dataframe(metrics, use_container_width=True)

    st.subheader("Outlier vs forward returns")
    if magnitude_col:
        melted = metrics.melt(
            id_vars=["ticker", "event_date", "magnitude"],
            value_vars=[c for c in metrics.columns if c.startswith("fwd_")],
            var_name="horizon",
            value_name="fwd_return",
        ).dropna(subset=["fwd_return", "magnitude"])
        if not melted.empty:
            fig_scatter = px.scatter(
                melted,
                x="magnitude",
                y="fwd_return",
                color="horizon",
                hover_data=["ticker", "event_date"],
                trendline="ols",
                title="Forward returns vs outlier magnitude",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            corr_rows = []
            for h in melted["horizon"].unique():
                sub = melted[melted["horizon"] == h][["magnitude", "fwd_return"]].dropna()
                corr = sub["magnitude"].corr(sub["fwd_return"]) if not sub.empty else np.nan
                corr_rows.append({"horizon": h, "corr": corr})
            st.dataframe(pd.DataFrame(corr_rows), use_container_width=True)
        else:
            st.info("No paired magnitude and forward returns to chart.")

    st.subheader("Event path (±7d)")
    if not windows.empty:
        ticker_choice = st.selectbox("Ticker", options=sorted(windows["ticker"].unique()))
        events_for_ticker = sorted(
            windows[windows["ticker"] == ticker_choice]["event_date"].drop_duplicates().dt.date.tolist()
        )
        event_choice = st.selectbox("Event date", options=events_for_ticker)
        scoped_window = windows[
            (windows["ticker"] == ticker_choice) & (windows["event_date"].dt.date == event_choice)
        ]
        fig_line = px.line(
            scoped_window,
            x="t_offset",
            y="rel_return",
            title=f"{ticker_choice} path around event ({event_choice})",
        )
        fig_line.add_vline(0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_line, use_container_width=True)

        mean_path = windows.groupby("t_offset")["rel_return"].mean().reset_index()
        fig_mean = px.line(mean_path, x="t_offset", y="rel_return", title="Average path across events")
        fig_mean.add_vline(0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_mean, use_container_width=True)
    else:
        st.info("No price window data to plot.")

    st.subheader("Raw price snippets")
    st.dataframe(prices.tail(20), use_container_width=True)

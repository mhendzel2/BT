import datetime as dt
from typing import Dict, Iterable, List, Optional, Tuple
import sys
import os

# Add parent directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import fetch_ib_data, fetch_yfinance_data


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Lowercase and remove underscores and spaces to standardize column names
    df.columns = [c.lower().replace("_", "").replace(" ", "") for c in df.columns]
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    return df


def _date_col(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["date", "trade_date", "asofdate", "timestamp"]:
        if candidate in df.columns:
            return candidate
    return None


def _read_csv(upload) -> Tuple[pd.DataFrame, Optional[str]]:
    if upload is None:
        return pd.DataFrame(), None
    try:
        df = pd.read_csv(upload)
    except Exception as exc:  # pragma: no cover - streamlit surface
        st.error(f"Could not read {getattr(upload, 'name', 'upload')}: {exc}")
        return pd.DataFrame(), None
    df = _normalize_cols(df)
    date_col = _date_col(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df, date_col


def _rolling_zscores(df: pd.DataFrame, group_col: str, date_col: str, cols: Iterable[str], window: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([group_col, date_col])
    for col in cols:
        if col not in out.columns:
            continue
        roll = out.groupby(group_col)[col].rolling(window, min_periods=max(5, window // 2))
        mean = roll.mean().reset_index(level=0, drop=True)
        std = roll.std().reset_index(level=0, drop=True).replace(0, np.nan)
        out[f"{col}_z"] = (out[col] - mean) / std
    return out


def _cross_sectional_z(df: pd.DataFrame, date_col: str, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    if date_col is None:
        return out
    for col in cols:
        if col not in out.columns:
            continue
        mean = out.groupby(date_col)[col].transform("mean")
        std = out.groupby(date_col)[col].transform("std").replace(0, np.nan)
        out[f"{col}_xsec_z"] = (out[col] - mean) / std
    return out


def _ensure_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].str.upper()
    return out


def _price_series(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["close", "last", "price", "underlying_price"]:
        if candidate in df.columns:
            return candidate
    return None


def prepare_stock_features(df: pd.DataFrame, date_col: Optional[str], window: int) -> Tuple[pd.DataFrame, Optional[str]]:
    if df.empty:
        return df, date_col
    df = _ensure_ticker(df)
    if date_col is None:
        date_col = "date"
        df[date_col] = pd.Timestamp(dt.date.today())

    price_col = _price_series(df)
    if price_col:
        df = df.sort_values([date_col, "ticker"])
        df["return_1d"] = df.groupby("ticker")[price_col].pct_change()

    volume_col = "totalvolume" if "totalvolume" in df.columns else "volume" if "volume" in df.columns else None
    if volume_col and "avg30volume" not in df.columns:
        df["avg30volume"] = df.groupby("ticker")[volume_col].rolling(30, min_periods=5).mean().reset_index(level=0, drop=True)

    feature_cols = [
        "return_1d",
        "totalvolume",
        "volume",
        "avg30volume",
        "callpremium",
        "putpremium",
        "bullishpremium",
        "bearishpremium",
        "callvolume",
        "putvolume",
        "putcallratio",
    ]
    df = _rolling_zscores(df, "ticker", date_col, feature_cols, window)
    df = _cross_sectional_z(df, date_col, feature_cols)
    return df, date_col


def _get_opt_type_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["option_type", "optiontype", "right", "type"]:
        if c in df.columns:
            return c
    return None


def _get_dte_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["dte", "days_to_expiry", "days_till_expiration"]:
        if c in df.columns:
            return c
    return None


def _find_ref_price(row, price_map: Dict[Tuple[str, pd.Timestamp], float], price_col: Optional[str], date_col: Optional[str]):
    if price_col and price_col in row and pd.notna(row[price_col]):
        return row[price_col]
    if date_col is None or "ticker" not in row:
        return np.nan
    key = (row["ticker"], row[date_col])
    return price_map.get(key, np.nan)


def aggregate_chain_oi(
    chain_df: pd.DataFrame,
    date_col: Optional[str],
    ref_prices: Dict[Tuple[str, pd.Timestamp], float],
    atm_band: float,
    max_dte: Optional[int],
) -> pd.DataFrame:
    if chain_df.empty:
        return pd.DataFrame()
    chain_df = _ensure_ticker(chain_df)
    if date_col is None:
        date_col = _date_col(chain_df) or "date"
        chain_df[date_col] = pd.Timestamp(dt.date.today())
    opt_col = _get_opt_type_col(chain_df)
    if opt_col is None or "oi_change" not in chain_df.columns:
        return pd.DataFrame()
    dte_col = _get_dte_col(chain_df)
    price_col = _price_series(chain_df)
    if "moneyness" in chain_df.columns:
        chain_df["atm_distance"] = chain_df["moneyness"].abs()
    elif "strike" in chain_df.columns:
        chain_df["atm_distance"] = chain_df.apply(
            lambda r: abs(r["strike"] - _find_ref_price(r, ref_prices, price_col, date_col))
            / _find_ref_price(r, ref_prices, price_col, date_col)
            if pd.notna(_find_ref_price(r, ref_prices, price_col, date_col)) and pd.notna(r.get("strike"))
            else np.nan,
            axis=1,
        )
    else:
        chain_df["atm_distance"] = np.nan
    mask = chain_df["atm_distance"].le(atm_band)
    if max_dte and dte_col and dte_col in chain_df:
        mask &= chain_df[dte_col].le(max_dte)
    scoped = chain_df[mask].copy()
    if scoped.empty:
        return pd.DataFrame()
    scoped["side"] = scoped[opt_col].astype(str).str.upper().str[0]
    agg = scoped.groupby([date_col, "ticker", "side"])["oi_change"].sum().reset_index()
    pivot = agg.pivot_table(index=[date_col, "ticker"], columns="side", values="oi_change", fill_value=0).reset_index()
    pivot = pivot.rename(columns={"C": "call_oi_change_atm", "P": "put_oi_change_atm"})
    if "call_oi_change_atm" not in pivot.columns:
        pivot["call_oi_change_atm"] = 0.0
    if "put_oi_change_atm" not in pivot.columns:
        pivot["put_oi_change_atm"] = 0.0
    return pivot


def merge_sources(
    stock_df: pd.DataFrame,
    chain_df: pd.DataFrame,
    date_col: Optional[str],
    atm_band: float,
    max_dte: int,
) -> pd.DataFrame:
    if stock_df.empty:
        return pd.DataFrame()
    price_col = _price_series(stock_df)
    ref_prices = {}
    if price_col and date_col:
        ref_prices = {(r["ticker"], r[date_col]): r[price_col] for _, r in stock_df.iterrows() if pd.notna(r.get(price_col))}
    chain_agg = aggregate_chain_oi(chain_df, date_col, ref_prices, atm_band, max_dte)
    merged = stock_df.merge(chain_agg, on=[date_col, "ticker"], how="left")
    merged["call_oi_change_atm"] = merged.get("call_oi_change_atm", 0).fillna(0)
    merged["put_oi_change_atm"] = merged.get("put_oi_change_atm", 0).fillna(0)
    return merged


def _signal_strength(values: List[float]) -> float:
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return 0.0
    strength = 1.0
    for v in vals:
        strength *= float(v)
    return strength


def build_signals(df: pd.DataFrame, date_col: str, params: Dict) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        ret_z = row.get("return_1d_z", np.nan)
        vol_z = row.get("totalvolume_z", row.get("volume_z", np.nan))
        call_prem_z = row.get("callpremium_z", np.nan)
        put_prem_z = row.get("putpremium_z", np.nan)
        call_vol_z = row.get("callvolume_z", np.nan)
        put_vol_z = row.get("putvolume_z", np.nan)
        pcr = row.get("putcallratio", np.nan)
        pcr_z = row.get("putcallratio_z", row.get("putcallratio_xsec_z", np.nan))
        call_oi = row.get("call_oi_change_atm", 0.0)
        put_oi = row.get("put_oi_change_atm", 0.0)
        avg_call_vol = row.get("avg30daycallvolume", row.get("callvolume_avg30", row.get("callvolume", np.nan)))
        vol_filter = pd.isna(avg_call_vol) or avg_call_vol >= params["min_call_liquidity"]
        base = {"ticker": row.get("ticker"), "date": row.get(date_col)}

        if vol_filter and ret_z is not None and call_prem_z is not None:
            if ret_z > params["momentum_z"] and call_prem_z > params["flow_z"] and call_oi > 0:
                strength = _signal_strength([ret_z, call_prem_z, vol_z or 1, call_oi or 1])
                records.append(
                    {
                        **base,
                        "signal": "Call Momentum Long",
                        "reason": "ret_z>thr, callprem_z>thr, call OI rising near ATM",
                        "signal_strength": strength,
                        "z_scores": {"return_1d_z": ret_z, "callpremium_z": call_prem_z, "volume_z": vol_z},
                    }
                )
            if ret_z < -params["momentum_z"] and put_prem_z > params["flow_z"] and put_oi > 0 and (pcr > params["pcr_min"] or (pcr_z is not None and pcr_z > 0)):
                strength = _signal_strength([abs(ret_z), put_prem_z, vol_z or 1, put_oi or 1])
                records.append(
                    {
                        **base,
                        "signal": "Put Momentum Short",
                        "reason": "ret_z<-thr, putprem_z>thr, put OI rising near ATM",
                        "signal_strength": strength,
                        "z_scores": {"return_1d_z": ret_z, "putpremium_z": put_prem_z, "volume_z": vol_z},
                    }
                )
        if vol_z is not None and call_vol_z is not None and vol_z > params["volume_breakout_z"] and call_vol_z > params["call_vol_z"]:
            strength = _signal_strength([vol_z, call_vol_z])
            records.append(
                {
                    **base,
                    "signal": "Volume Breakout",
                    "reason": "total volume and call volume extreme",
                    "signal_strength": strength,
                    "z_scores": {"volume_z": vol_z, "callvolume_z": call_vol_z},
                }
            )

        if pd.notna(row.get("nextearningsdate")):
            try:
                earnings_dt = pd.to_datetime(row["nextearningsdate"])
                dte = (earnings_dt.normalize() - pd.to_datetime(row[date_col]).normalize()).days if pd.notna(row.get(date_col)) else np.nan
            except Exception:
                dte = np.nan
            if pd.notna(dte) and params["earn_min_dte"] <= dte <= params["earn_max_dte"]:
                ertimes = str(row.get("ertimes", "")).lower()
                if call_prem_z and call_prem_z > params["earn_call_z"]:
                    records.append(
                        {
                            **base,
                            "signal": "Call Runup Long",
                            "reason": f"Earnings in {dte}d, callpremium_z high",
                            "signal_strength": _signal_strength([call_prem_z, 1 + (0.1 if "pre" in ertimes or "post" in ertimes else 0)]),
                            "z_scores": {"callpremium_z": call_prem_z},
                        }
                    )
                if put_prem_z and put_prem_z > params["earn_put_z"] and row.get("week52high") and row.get("close") and row["close"] >= 0.98 * row["week52high"]:
                    records.append(
                        {
                            **base,
                            "signal": "Put Fade Short",
                            "reason": f"Earnings in {dte}d, putpremium_z high near 52w high",
                            "signal_strength": _signal_strength([put_prem_z]),
                            "z_scores": {"putpremium_z": put_prem_z},
                        }
                    )
                iv_1w = row.get("iv30d1w")
                iv_1m = row.get("iv30d1m")
                if pd.notna(iv_1w) and pd.notna(iv_1m) and iv_1w > (1 + params["skew_gap"]) * iv_1m:
                    call_ask = row.get("callvolumeaskside", np.nan)
                    call_bid = row.get("callvolumebidside", np.nan)
                    if pd.isna(call_ask) or pd.isna(call_bid) or call_ask > call_bid:
                        records.append(
                            {
                                **base,
                                "signal": "Skew Reversion",
                                "reason": f"Earnings in {dte}d, front IV rich vs 1m",
                                "signal_strength": _signal_strength([iv_1w / iv_1m]),
                                "z_scores": {},
                            }
                        )
    return pd.DataFrame(records)


def _cluster_anomalies(df: pd.DataFrame, feature_cols: List[str], n_clusters: int) -> pd.DataFrame:
    try:
        from sklearn.cluster import KMeans
    except Exception:
        st.warning("scikit-learn not installed; skipping clustering.")
        return pd.DataFrame()
    scoped = df.dropna(subset=feature_cols)
    if scoped.empty:
        return pd.DataFrame()
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    scoped = scoped.copy()
    scoped["cluster"] = model.fit_predict(scoped[feature_cols])
    counts = scoped["cluster"].value_counts().sort_values()
    small_clusters = counts[counts <= counts.median()].index.tolist()
    scoped["cluster_flagged"] = scoped["cluster"].isin(small_clusters)
    return scoped[["ticker", feature_cols[0]] + ["cluster", "cluster_flagged"]]


import os
import glob
from db_utils import get_engine, init_db, ingest_folder, load_data_from_db, DEFAULT_DB_URL

def _load_local_files(folder: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_df = pd.DataFrame()
    chain_df = pd.DataFrame()
    hot_df = pd.DataFrame()
    eod_df = pd.DataFrame()
    
    # Handle quotes and file paths
    folder = folder.strip('"').strip("'")
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    
    if not os.path.exists(folder):
        st.warning(f"Folder {folder} does not exist.")
        return stock_df, chain_df, hot_df, eod_df

    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        st.warning(f"No CSV files found in {os.path.abspath(folder)}.")
        return stock_df, chain_df, hot_df, eod_df

    st.info(f"Found {len(files)} files in {folder}. Attempting to categorize and merge...")
    with st.expander("See found files"):
        st.write([os.path.basename(f) for f in files])

    stock_frames = []
    chain_frames = []
    hot_frames = []
    eod_frames = []
    
    for f in files:
        try:
            # Try reading with default settings first
            try:
                df = pd.read_csv(f)
                # If only one column, it might be tab separated or semicolon
                if len(df.columns) <= 1:
                     df = pd.read_csv(f, sep=None, engine='python')
            except:
                # Fallback to python engine auto-detection
                df = pd.read_csv(f, sep=None, engine='python')

            df = _normalize_cols(df)
            date_col = _date_col(df)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            
            # Heuristic to identify file type
            cols = set(df.columns)
            if "callpremium" in cols or "putpremium" in cols:
                stock_frames.append(df)
            elif ("oichange" in cols or "oi_change" in cols) and "strike" in cols:
                chain_frames.append(df)
            elif "optionsymbol" in cols and "tapetime" in cols:
                hot_frames.append(df)
            elif "tradecode" in cols and "nbboask" in cols:
                eod_frames.append(df)
            else:
                st.warning(f"Skipping {os.path.basename(f)}: Missing required columns. Found: {list(cols)}")
            
        except Exception as e:
            st.error(f"Error reading {f}: {e}")

    if stock_frames:
        stock_df = pd.concat(stock_frames, ignore_index=True)
        st.success(f"Loaded {len(stock_frames)} stock screener files.")
    else:
        st.warning("No stock screener files identified (looking for 'callpremium' or 'putpremium' columns).")

    if chain_frames:
        chain_df = pd.concat(chain_frames, ignore_index=True)
        st.success(f"Loaded {len(chain_frames)} chain OI files.")

    if hot_frames:
        hot_df = pd.concat(hot_frames, ignore_index=True)
        st.success(f"Loaded {len(hot_frames)} hot chain files.")

    if eod_frames:
        eod_df = pd.concat(eod_frames, ignore_index=True)
        st.success(f"Loaded {len(eod_frames)} EOD report files.")
        
    return stock_df, chain_df, hot_df, eod_df

st.title("Signal Scanner: Flow, Momentum, Earnings")
st.caption(
    "Upload daily files to surface cross-sectional outliers, flow-backed momentum setups, and pre-earnings plays. "
    "Rolling z-scores standardize features for ranking."
)

with st.sidebar:
    st.subheader("Inputs")
    input_method = st.radio("Input Method", ["Database", "Upload Files", "Load from Local Folder"])
    
    stock_upload = None
    chain_upload = None
    hot_upload = None
    eod_upload = None
    local_folder = "sample_data"
    db_url = DEFAULT_DB_URL

    if input_method == "Database":
        db_url = st.text_input("Database URL", value=DEFAULT_DB_URL, type="password")
        local_folder = st.text_input("Folder to Ingest", value="sample_data", help="Folder containing CSVs to add to DB.")
        if st.button("Ingest/Update Database"):
            try:
                engine = get_engine(db_url)
                init_db(engine)
                count, errors = ingest_folder(local_folder, engine)
                st.success(f"Imported {count} new files.")
                if errors:
                    st.error(f"Encountered {len(errors)} errors.")
                    with st.expander("See errors"):
                        st.write(errors)
            except Exception as e:
                st.error(f"Database error: {e}")

    elif input_method == "Upload Files":
        stock_upload = st.file_uploader("Stock screener CSV", type=["csv"], help="Includes price/volume/premium and nextearningsdate fields.")
        chain_upload = st.file_uploader("Chain OI changes CSV", type=["csv"], help="Option-level OI change with strike/dte.")
        hot_upload = st.file_uploader("Hot chains CSV (optional)", type=["csv"])
        eod_upload = st.file_uploader("EOD report CSV (optional)", type=["csv"])
    else:
        local_folder = st.text_input("Folder Path", value="sample_data", help="Absolute path to folder containing CSV files.")
    
    st.subheader("Settings")
    lookback = st.slider("Rolling window (days)", min_value=10, max_value=60, value=20, step=5)
    atm_band = st.slider("ATM band (as fraction of spot)", 0.01, 0.25, 0.10, step=0.01)
    max_dte = st.slider("Max DTE for flow filters", 1, 30, 10)
    min_call_liquidity = st.number_input("Min avg 30d call volume", value=1000, min_value=0, step=100)
    momentum_z = st.slider("Momentum |z| threshold", 1.5, 4.0, 2.0, step=0.1)
    flow_z = st.slider("Flow premium z threshold", 1.5, 4.0, 2.0, step=0.1)
    volume_breakout_z = st.slider("Volume breakout z threshold", 2.0, 6.0, 3.0, step=0.1)
    call_vol_z = st.slider("Call volume z threshold", 1.0, 5.0, 2.0, step=0.1)
    pcr_min = st.slider("Put/Call ratio min (puts)", 0.5, 3.0, 1.5, step=0.1)
    earn_min_dte, earn_max_dte = st.slider("Earnings window (days)", 0, 10, (1, 5))
    earn_call_z = st.slider("Earnings callpremium z", 1.5, 4.0, 2.5, step=0.1)
    earn_put_z = st.slider("Earnings putpremium z", 1.0, 4.0, 2.0, step=0.1)
    skew_gap = st.slider("Skew rich threshold (1w vs 1m IV)", 0.05, 0.50, 0.20, step=0.01)
    do_cluster = st.checkbox("Cluster anomalies (k-means)", value=False)
    n_clusters = st.slider("Clusters", 2, 6, 3) if do_cluster else 0

run = st.button("Run analysis")

if run:
    if input_method == "Database":
        try:
            engine = get_engine(db_url)
            stock_df, chain_df, hot_df, eod_df = load_data_from_db(engine)
            stock_date_col = "date"
            chain_date_col = "date"
            hot_date_col = "date"
            eod_date_col = "date"
        except Exception as e:
            st.error(f"Failed to load from database: {e}")
            st.stop()
    elif input_method == "Upload Files":
        stock_df, stock_date_col = _read_csv(stock_upload)
        chain_df, chain_date_col = _read_csv(chain_upload)
        hot_df, hot_date_col = _read_csv(hot_upload)
        eod_df, eod_date_col = _read_csv(eod_upload)
    else:
        stock_df, chain_df, hot_df, eod_df = _load_local_files(local_folder)
        stock_date_col = _date_col(stock_df) if not stock_df.empty else None
        chain_date_col = _date_col(chain_df) if not chain_df.empty else None
        hot_date_col = _date_col(hot_df) if not hot_df.empty else None
        eod_date_col = _date_col(eod_df) if not eod_df.empty else None

    if stock_df.empty:
        msg = "Stock screener data is required."
        if input_method == "Database":
            msg += " Ensure you have ingested data into the database."
        else:
            msg += f" Please upload a file or ensure '{local_folder}' contains CSVs with 'callpremium' or 'putpremium' columns."
        st.error(msg)
        st.stop()

    stock_df, date_col = prepare_stock_features(stock_df, stock_date_col, lookback)
    merged = merge_sources(stock_df, chain_df, date_col, atm_band, max_dte)

    params = {
        "momentum_z": momentum_z,
        "flow_z": flow_z,
        "volume_breakout_z": volume_breakout_z,
        "call_vol_z": call_vol_z,
        "pcr_min": pcr_min,
        "min_call_liquidity": min_call_liquidity,
        "earn_min_dte": earn_min_dte,
        "earn_max_dte": earn_max_dte,
        "earn_call_z": earn_call_z,
        "earn_put_z": earn_put_z,
        "skew_gap": skew_gap,
    }
    signals = build_signals(merged, date_col, params)

    st.subheader("Ranked Signals")
    if signals.empty:
        st.info("No signals met the thresholds. Loosen z-score or DTE settings.")
    else:
        signals = signals.sort_values("signal_strength", ascending=False)
        st.dataframe(signals[["date", "ticker", "signal", "signal_strength", "reason"]].head(50), use_container_width=True)

        st.subheader("Top Signal Detail")
        top = signals.head(15)
        fig = px.bar(top, x="ticker", y="signal_strength", color="signal", hover_data=["reason"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("IBKR Performance Analysis")
        st.caption("Fetch historical data to analyze the trajectory of these signals.")
        
        if st.button("Analyze Signals with IBKR"):
            with st.spinner("Fetching market data from IBKR..."):
                unique_tickers = signals["ticker"].unique().tolist()
                # Fetch data from earliest signal date to today
                start_date = pd.to_datetime(signals["date"]).min().date()
                end_date = dt.date.today()
                
                try:
                    # Try IBKR first
                    market_data = fetch_ib_data(unique_tickers, start_date, end_date)
                except Exception as e:
                    st.error(f"IBKR connection failed: {e}. Falling back to Yahoo Finance.")
                    market_data = fetch_yfinance_data(unique_tickers, start_date, end_date)
                
                if market_data and not market_data["close"].empty:
                    perf_results = []
                    close_prices = market_data["close"]
                    
                    for _, row in signals.iterrows():
                        ticker = row["ticker"]
                        signal_date = pd.to_datetime(row["date"])
                        signal_type = row["signal"]
                        
                        if ticker not in close_prices.columns:
                            continue
                            
                        prices = close_prices[ticker].dropna()
                        if prices.empty:
                            continue

                        # Find closest date on or after signal date
                        # Use searchsorted to find insertion point
                        idx = prices.index.searchsorted(signal_date)
                        
                        if idx < len(prices):
                            start_price = prices.iloc[idx]
                            current_price = prices.iloc[-1]
                            
                            # Calculate return
                            ret = (current_price - start_price) / start_price
                            
                            # Adjust for short signals
                            if "Short" in signal_type or "Put" in signal_type:
                                ret = -ret
                                
                            perf_results.append({
                                "ticker": ticker,
                                "signal_date": signal_date.date(),
                                "signal": signal_type,
                                "start_price": start_price,
                                "current_price": current_price,
                                "return": ret,
                                "days_held": (prices.index[-1] - signal_date).days
                            })
                    
                    if perf_results:
                        perf_df = pd.DataFrame(perf_results)
                        st.write("### Performance Trajectory")
                        
                        # Color code returns
                        def color_return(val):
                            color = 'green' if val > 0 else 'red'
                            return f'color: {color}'
                        
                        st.dataframe(
                            perf_df.style.format({
                                "start_price": "{:.2f}",
                                "current_price": "{:.2f}",
                                "return": "{:.2%}"
                            }).applymap(color_return, subset=['return']),
                            use_container_width=True
                        )
                        
                        # Plot trajectory
                        fig_perf = px.bar(
                            perf_df, 
                            x="ticker", 
                            y="return", 
                            color="signal", 
                            title="Return Since Signal",
                            hover_data=["days_held", "start_price", "current_price"]
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                        
                        # Outlier Analysis on Performance
                        st.write("### Leading Indicators Analysis")
                        st.caption("Correlation between signal strength and realized return.")
                        
                        # Merge performance back with signals to see if strength predicted return
                        merged_perf = pd.merge(
                            signals, 
                            perf_df[["ticker", "signal_date", "return"]], 
                            left_on=["ticker", "date"], 
                            right_on=["ticker", "signal_date"]
                        )
                        
                        if not merged_perf.empty:
                            fig_corr = px.scatter(
                                merged_perf, 
                                x="signal_strength", 
                                y="return", 
                                color="signal", 
                                hover_data=["ticker", "reason"],
                                trendline="ols",
                                title="Signal Strength vs. Realized Return"
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.warning("Could not calculate performance for any signals (missing price data).")
                else:
                    st.error("No market data returned.")

    st.subheader("Outlier Heatmap (cross-sectional z)")
    heat_cols = [c for c in merged.columns if c.endswith("_xsec_z")]
    if heat_cols:
        latest_date = merged[date_col].max()
        latest = merged[merged[date_col] == latest_date].copy()
        melted = latest.melt(id_vars=["ticker"], value_vars=heat_cols, var_name="feature", value_name="z")
        pivot = melted.pivot(index="feature", columns="ticker", values="z")
        fig = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="RdBu",
            origin="lower",
        )
        vals = melted["z"].dropna()
        if not vals.empty:
            hi = np.percentile(vals, 95)
            lo = np.percentile(vals, 5)
            extremes = melted[(melted["z"] >= hi) | (melted["z"] <= lo)]
            for _, r in extremes.iterrows():
                fig.add_annotation(
                    x=r["ticker"],
                    y=r["feature"],
                    text="★",
                    showarrow=False,
                    font=dict(color="black", size=10),
                )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cross-sectional z-scores available to plot.")

    if do_cluster:
        feat_cols = [c for c in ["return_1d_z", "callpremium_z", "putpremium_z", "totalvolume_z", "callvolume_z"] if c in merged.columns]
        cluster_df = _cluster_anomalies(merged, feat_cols, n_clusters) if feat_cols else pd.DataFrame()
        if not cluster_df.empty:
            st.subheader("Clustering Highlights")
            st.dataframe(cluster_df.head(30), use_container_width=True)
        else:
            st.info("Not enough data for clustering.")

    st.subheader("Cumulative Flow (5-day rolling net premium)")
    flow_df = merged.copy()
    if not flow_df.empty and date_col in flow_df.columns and {"ticker"}.issubset(flow_df.columns):
        flow_df["net_premium"] = flow_df.get("netcallpremium", flow_df.get("callpremium", 0)) - flow_df.get(
            "netputpremium", flow_df.get("putpremium", 0)
        )
        flow_df = flow_df.dropna(subset=[date_col, "ticker"]).sort_values([ "ticker", date_col])
        flow_df["net_roll5"] = (
            flow_df.groupby("ticker")["net_premium"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
        )
        tickers = sorted(flow_df["ticker"].dropna().unique().tolist())
        default_sel = tickers[:5]
        sel = st.multiselect("Tickers to chart", options=tickers, default=default_sel)
        scoped = flow_df[flow_df["ticker"].isin(sel)]
        if not scoped.empty:
            fig = px.line(scoped, x=date_col, y="net_roll5", color="ticker", title="5d rolling net premium (calls - puts)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one ticker with flow data.")
    else:
        st.info("Flow chart needs ticker, date, and premium columns (call/put).")

    st.subheader("Earnings Calendar Scatter")
    if "nextearningsdate" in merged.columns and merged["nextearningsdate"].notna().any():
        earn = merged.copy()
        earn["earn_date"] = pd.to_datetime(earn["nextearningsdate"], errors="coerce")
        earn["dte_earnings"] = (earn["earn_date"].dt.normalize() - pd.to_datetime(earn[date_col]).dt.normalize()).dt.days
        earn = earn.dropna(subset=["dte_earnings"])
        id_vars = [date_col, "ticker", "dte_earnings"]
        if "totalvolume" in earn.columns:
            id_vars.append("totalvolume")
        if "ertimes" in earn.columns:
            id_vars.append("ertimes")
        melted = earn.melt(
            id_vars=id_vars,
            value_vars=[c for c in ["callpremium_z", "putpremium_z"] if c in earn.columns],
            var_name="side",
            value_name="premium_z",
        )
        if not melted.empty:
            fig = px.scatter(
                melted,
                x="dte_earnings",
                y="premium_z",
                color="side",
                size="totalvolume" if "totalvolume" in melted.columns else None,
                hover_data=["ticker", "ertimes"] if "ertimes" in melted.columns else ["ticker"],
                title="Premium z-scores vs days to earnings",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need callpremium_z or putpremium_z to plot earnings scatter.")
    else:
        st.info("No earnings dates found; add nextearningsdate to stock screener file.")

    st.subheader("Merged Data (preview)")
    st.dataframe(merged.head(100), use_container_width=True)

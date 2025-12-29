import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from pathlib import Path


st.title("OI Outlier Scanner")
st.caption(
    "Detect unusual open interest changes with multiple statistical filters, pre-earnings screens, "
    "and quick visuals for manipulation flags."
)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Lowercase and remove underscores and spaces to standardize column names
    df.columns = [c.lower().replace("_", "").replace(" ", "") for c in df.columns]
    return df


def _read_many(items):
    frames = []
    for item in items:
        try:
            df = pd.read_csv(item)
            # If only one column, it might be tab separated or semicolon
            if len(df.columns) <= 1:
                    df = pd.read_csv(item, sep=None, engine='python')
            df = _normalize_cols(df)
            frames.append(df)
        except Exception as exc:  # pragma: no cover - Streamlit surface
            name = item if isinstance(item, (str, Path)) else getattr(item, "name", "file")
            st.error(f"Could not read {name}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    return df


def _parse_dates(df: pd.DataFrame, curr_col: str, earn_col: str):
    if curr_col in df.columns:
        df[curr_col] = pd.to_datetime(df[curr_col].astype(str).str.strip(), errors="coerce")
    if earn_col in df.columns:
        df[earn_col] = pd.to_datetime(df[earn_col].astype(str).str.strip(), errors="coerce")
    if curr_col in df.columns and earn_col in df.columns:
        df["days_to_earnings"] = (df[earn_col] - df[curr_col]).dt.days
    return df


def zscore_outliers(series: pd.Series, threshold: float):
    clean = series.dropna()
    if clean.std() == 0 or clean.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    z = (clean - clean.mean()) / clean.std()
    abs_z = z.abs()
    return abs_z, abs_z[abs_z > threshold].index


def iqr_bounds(series: pd.Series, multiplier: float):
    clean = series.dropna()
    if clean.empty:
        return (np.nan, np.nan)
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper


with st.sidebar:
    st.subheader("Inputs")
    oi_uploads = st.file_uploader("chain-oi-changes CSVs", type=["csv"], accept_multiple_files=True)
    dir_path = st.text_input("Directory with outlier CSVs", value="", help="Optional: load all CSVs from this folder.")
    hot_upload = st.file_uploader("hot-chains CSV (optional)", type=["csv"])
    eod_upload = st.file_uploader("dp-eod-report CSV (optional)", type=["csv"])

    st.subheader("Settings")
    z_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.0, step=0.1)
    iqr_mult = st.slider("IQR multiplier", 0.5, 3.0, 1.5, step=0.1)
    manip_pct = st.slider("Min % of total chain", 0.05, 0.5, 0.20, step=0.01)
    manip_quantile = st.slider("Min OI Change quantile", 0.80, 0.99, 0.95, step=0.01)
    pre_earn_days = st.slider("Pre-earnings window (days)", 3, 30, 14)
    top_n = st.slider("Top candidates to display", 5, 30, 10)

run = st.button("Run analysis")

if run:
    inputs = list(oi_uploads or [])
    if dir_path.strip():
        folder = Path(dir_path).expanduser()
        if folder.is_dir():
            found = sorted(folder.glob("*.csv"))
            inputs.extend(found)
            st.info(f"Loaded {len(found)} CSV files from {folder}")
        else:
            st.warning(f"Directory not found: {folder}")

    df_oi = _read_many(inputs)
    df_hot = _read_many([hot_upload] if hot_upload else [])
    df_eod = _read_many([eod_upload] if eod_upload else [])

    if df_oi.empty:
        st.error("Please upload the chain-oi-changes CSV.")
        st.stop()

    # Ensure we have the right column names
    if "oichange" not in df_oi.columns and "oidiff" in df_oi.columns:
        df_oi = df_oi.rename(columns={"oidiff": "oichange"})
    
    if "underlyingsymbol" not in df_oi.columns:
        if "ticker" in df_oi.columns:
            df_oi = df_oi.rename(columns={"ticker": "underlyingsymbol"})
        elif "symbol" in df_oi.columns:
            df_oi = df_oi.rename(columns={"symbol": "underlyingsymbol"})
    
    # Fallback for date column
    date_col = "asofdate"
    if "currdate" in df_oi.columns:
        date_col = "currdate"
    elif "date" in df_oi.columns:
        date_col = "date"

    df_oi = _to_numeric(df_oi, ["oichange", "stockprice", "percentageoftotal", "dte"])
    df_oi = _parse_dates(df_oi, date_col, "nextearningsdate")
    oi_clean = df_oi["oichange"].dropna()

    abs_z, z_idx = zscore_outliers(oi_clean, z_thresh)
    df_oi.loc[abs_z.index, "z_score"] = abs_z
    z_outliers = df_oi.loc[z_idx].copy().sort_values("oichange", ascending=False)

    lower_bound, upper_bound = iqr_bounds(oi_clean, iqr_mult)
    df_oi["is_iqr_outlier"] = (df_oi["oichange"] < lower_bound) | (df_oi["oichange"] > upper_bound)
    iqr_outliers = df_oi[df_oi["is_iqr_outlier"]].copy().sort_values("oichange", ascending=False)

    pre_earn = df_oi[df_oi["days_to_earnings"].notna() & (df_oi["days_to_earnings"] < pre_earn_days)].copy()
    threshold = oi_clean.quantile(manip_quantile) if not oi_clean.empty else np.nan
    manip_candidates = pre_earn[
        (pre_earn["percentageoftotal"] > manip_pct) & (pre_earn["oichange"] > threshold)
    ].copy()
    if not manip_candidates.empty:
        manip_candidates["manip_score"] = (
            manip_candidates["oichange"]
            * manip_candidates["percentageoftotal"]
            * (1 / manip_candidates["days_to_earnings"].clip(lower=1))
        )
        top_manip = manip_candidates.nlargest(top_n, "manip_score")
    else:
        top_manip = pd.DataFrame(columns=["underlyingsymbol", "manip_score"])

    st.subheader("Summary")
    summary = pd.DataFrame(
        {
            "Method": ["Z-Score", "IQR", "Pre-Event Manip"],
            "Count": [len(z_outliers), len(iqr_outliers), len(top_manip)],
            "Top Symbol": [
                z_outliers["underlyingsymbol"].iloc[0] if not z_outliers.empty else "N/A",
                iqr_outliers["underlyingsymbol"].iloc[0] if not iqr_outliers.empty else "N/A",
                top_manip["underlyingsymbol"].iloc[0] if not top_manip.empty else "N/A",
            ],
            "Max OI Change": [
                z_outliers["oichange"].max() if not z_outliers.empty else 0,
                iqr_outliers["oichange"].max() if not iqr_outliers.empty else 0,
                top_manip["oichange"].max() if not top_manip.empty else 0,
            ],
        }
    )
    st.dataframe(summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Z-Score Outliers")
        st.dataframe(z_outliers[["underlyingsymbol", "plainoptionsymbol", "oichange", "z_score", "days_to_earnings"]].head(top_n))
    with col2:
        st.write("IQR Outliers")
        st.dataframe(iqr_outliers[["underlyingsymbol", "oichange", "percentageoftotal"]].head(top_n))

    st.write("Top Manipulation Candidates")
    st.dataframe(
        top_manip[
            ["underlyingsymbol", "oichange", "percentageoftotal", "days_to_earnings", "manip_score"]
        ].head(top_n),
        use_container_width=True,
    )

    st.subheader("Visualizations")
    if not oi_clean.empty:
        fig_hist = px.histogram(oi_clean, nbins=50, title="OI Changes Distribution")
        if not np.isnan(lower_bound):
            fig_hist.add_vline(lower_bound, line_dash="dash", line_color="red")
        if not np.isnan(upper_bound):
            fig_hist.add_vline(upper_bound, line_dash="dash", line_color="red")
        if not z_outliers.empty:
            fig_hist.add_scatter(
                x=z_outliers["oichange"],
                y=[0] * len(z_outliers),
                mode="markers",
                marker=dict(color="orange", size=10),
                name="Z outliers",
            )
        st.plotly_chart(fig_hist, use_container_width=True)

    if not top_manip.empty:
        fig_bar = px.bar(
            top_manip.head(top_n),
            x="underlyingsymbol",
            y="manip_score",
            hover_data=["oichange", "percentageoftotal", "days_to_earnings"],
            title="Top Manipulation Candidates",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    symbol_corr = []
    if not top_manip.empty:
        for symbol in top_manip["underlyingsymbol"].dropna().unique():
            sym = df_oi[df_oi["underlyingsymbol"] == symbol].dropna(subset=["oichange", "stockprice"])
            if len(sym) > 5 and sym["oichange"].std() > 0 and sym["stockprice"].std() > 0:
                corr = np.corrcoef(sym["oichange"], sym["stockprice"])[0, 1]
                symbol_corr.append({"symbol": symbol, "corr": corr, "n": len(sym)})
    corr_df = pd.DataFrame(symbol_corr)
    if not corr_df.empty:
        fig_corr = px.bar(corr_df, x="symbol", y="corr", title="OI vs Stock Price Correlation (suspect symbols)")
        st.plotly_chart(fig_corr, use_container_width=True)

    if not pre_earn.empty:
        fig_scatter = px.scatter(
            pre_earn.dropna(subset=["oichange", "days_to_earnings"]),
            x="days_to_earnings",
            y="oichange",
            color="percentageoftotal",
            size="percentageoftotal",
            hover_data=["underlyingsymbol", "plainoptionsymbol"] if "plainoptionsymbol" in pre_earn.columns else ["underlyingsymbol"],
            title="Pre-earnings OI spikes",
        )
        fig_scatter.add_vline(pre_earn_days, line_dash="dash", line_color="red")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Export")
    st.download_button("Download z-score outliers", data=z_outliers.to_csv(index=False), file_name="zscore_outliers.csv")
    st.download_button("Download IQR outliers", data=iqr_outliers.to_csv(index=False), file_name="iqr_outliers.csv")
    st.download_button("Download manipulation candidates", data=top_manip.to_csv(index=False), file_name="manipulation_candidates.csv")

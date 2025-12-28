import datetime as dt
import os
from typing import Dict, Iterable, Optional

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "GOOGL",
    "NVDA",
    "JPM",
    "UNH",
    "XOM",
    "JNJ",
    "V",
    "PG",
    "HD",
    "MA",
    "AVGO",
    "CVX",
    "ABBV",
    "PEP",
    "COST",
    "ADBE",
]


def _normalize_inputs(tickers: Iterable[str], start: dt.date, end: dt.date):
    tickers = sorted(set([t.upper().strip() for t in tickers if t]))
    if not tickers:
        tickers = DEFAULT_TICKERS
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    return tickers, start, end


def _extract_price_frames(raw: pd.DataFrame, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    tickers = list(tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Adj Close"]
        high = raw["High"]
        low = raw["Low"]
    else:
        name = tickers[0] if tickers else "Price"
        close = raw[["Adj Close"]].rename(columns={"Adj Close": name})
        high = raw[["High"]].rename(columns={"High": name})
        low = raw[["Low"]].rename(columns={"Low": name})
    close = close.dropna(how="all")
    high = high.loc[close.index]
    low = low.loc[close.index]
    return {"close": close, "high": high, "low": low}


def fetch_yfinance_data(
    tickers: Iterable[str],
    start: dt.date,
    end: dt.date,
    benchmark: str = "SPY",
) -> Dict[str, pd.DataFrame]:
    tickers, start, end = _normalize_inputs(tickers, start, end)
    symbols = sorted(set(tickers + [benchmark]))
    raw = yf.download(symbols, start=start, end=end, auto_adjust=False, progress=False)
    frames = _extract_price_frames(raw, symbols)
    frames["benchmark"] = benchmark
    return frames


def fetch_ib_data(
    tickers: Iterable[str],
    start: dt.date,
    end: dt.date,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 99,
    benchmark: str = "SPY",
) -> Dict[str, pd.DataFrame]:
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    try:
        from ib_insync import IB, Stock, util
    except ImportError as exc:
        raise ImportError("ib_insync is required for IBKR data") from exc

    tickers, start, end = _normalize_inputs(tickers, start, end)
    symbols = sorted(set(tickers + [benchmark]))
    ib = IB()
    ib.connect(host, port, clientId=client_id)

    close_frames = []
    high_frames = []
    low_frames = []

    duration_days = max(30, int((pd.to_datetime(end) - pd.to_datetime(start)).days) + 1)
    if duration_days > 365:
        years = int(duration_days / 365) + 1
        duration_str = f"{years} Y"
    else:
        duration_str = f"{duration_days} D"

    for symbol in symbols:
        contract = Stock(symbol, "SMART", "USD")
        try:
            ib.qualifyContracts(contract)
        except Exception as e:
            print(f"Error qualifying contract for {symbol}: {e}")
            continue

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting="1 day",
            whatToShow="ADJUSTED_LAST",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            print(f"No bars returned for {symbol}")
            continue
        frame = util.df(bars).set_index("date")[["close", "high", "low"]]
        close_frames.append(frame["close"].rename(symbol))
        high_frames.append(frame["high"].rename(symbol))
        low_frames.append(frame["low"].rename(symbol))

    if close_frames:
        close = pd.concat(close_frames, axis=1)
        high = pd.concat(high_frames, axis=1).loc[close.index]
        low = pd.concat(low_frames, axis=1).loc[close.index]
    else:
        close = high = low = pd.DataFrame()

    ib.disconnect()
    return {"close": close, "high": high, "low": low, "benchmark": benchmark}


def fetch_csv_data(
    tickers: Iterable[str],
    start: dt.date,
    end: dt.date,
    benchmark: str = "SPY",
    folder: str = "sample_data",
) -> Dict[str, pd.DataFrame]:
    tickers, start, end = _normalize_inputs(tickers, start, end)
    symbols = sorted(set(tickers + [benchmark]))

    close_frames = []
    high_frames = []
    low_frames = []

    for symbol in symbols:
        # Try various extensions or exact match
        path = os.path.join(folder, f"{symbol}.csv")
        if not os.path.exists(path):
            # Try lowercase
            path_lower = os.path.join(folder, f"{symbol.lower()}.csv")
            if os.path.exists(path_lower):
                path = path_lower
            else:
                print(f"File not found: {path}")
                continue

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            df = df.sort_index()
            # Filter by date
            mask = (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
            df = df.loc[mask]

            # Handle column names (case insensitive)
            df.columns = [c.strip() for c in df.columns]
            col_map = {c.lower(): c for c in df.columns}
            
            # Close
            if "adj close" in col_map:
                c = df[col_map["adj close"]]
            elif "close" in col_map:
                c = df[col_map["close"]]
            else:
                print(f"No close price column in {path}")
                continue

            # High
            if "high" in col_map:
                h = df[col_map["high"]]
            else:
                h = c

            # Low
            if "low" in col_map:
                l = df[col_map["low"]]
            else:
                l = c

            close_frames.append(c.rename(symbol))
            high_frames.append(h.rename(symbol))
            low_frames.append(l.rename(symbol))

        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

    if close_frames:
        close = pd.concat(close_frames, axis=1)
        high = pd.concat(high_frames, axis=1).reindex(close.index)
        low = pd.concat(low_frames, axis=1).reindex(close.index)
    else:
        close = high = low = pd.DataFrame()

    return {"close": close, "high": high, "low": low, "benchmark": benchmark}


def fetch_market_data(
    tickers: Iterable[str],
    start: dt.date,
    end: dt.date,
    source: str = "yfinance",
    benchmark: str = "SPY",
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    if source == "ibkr":
        return fetch_ib_data(tickers, start, end, benchmark=benchmark, **kwargs)
    elif source == "csv":
        return fetch_csv_data(tickers, start, end, benchmark=benchmark, **kwargs)
    return fetch_yfinance_data(tickers, start, end, benchmark=benchmark)

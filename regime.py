import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    high_low = high - low
    high_close = (high - prev_close).abs()
    low_close = (low - prev_close).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr


def choppiness_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    tr = true_range(high, low, close)
    atr_sum = tr.rolling(period).sum()
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    range_ = (highest_high - lowest_low).replace(0, np.nan)
    chop = 100 * np.log10(atr_sum / range_) / np.log10(period)
    return chop


def classify_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    trend_threshold: float = 55.0,
) -> pd.DataFrame:
    chop = choppiness_index(high, low, close, period=period)
    regime = np.where(chop <= trend_threshold, "trending", "consolidating")
    df = pd.DataFrame(
        {
            "chop": chop,
            "regime": regime,
            "trending": chop <= trend_threshold,
            "consolidating": chop > trend_threshold,
        }
    )
    return df.dropna()


def regime_mask(regime_frame: pd.DataFrame, name: str) -> pd.Series:
    col = name.lower()
    if col not in regime_frame.columns:
        raise ValueError(f"Unknown regime {name}")
    return regime_frame[col]

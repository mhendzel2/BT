# Momentum Optimizer (IBKR / Yahoo Finance)

Streamlit GUI to explore how lookback and holding periods perform in trending versus consolidating markets. Uses IBKR TWS/Gateway data via `ib_insync` or Yahoo Finance as a fallback. Regime detection is based on the choppiness index of SPY, letting you optimize parameters separately for each regime with heatmaps and equity curves.

## Features
- Interactive GUI for lookback and holding-period grid search
- Market regime split (trending vs consolidating) using choppiness index
- Heatmaps of CAGR per regime and equity curves for best parameter sets
- Supports IBKR TWS/Gateway or Yahoo Finance data
- Default S&P 500-heavy universe for quick experimentation

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Using IBKR data
1. Launch TWS or IB Gateway with API enabled (allow connections and set trusted host/port).
2. In the sidebar, choose `ibkr` as the data source and set host/port/client id if different from defaults (127.0.0.1 / 7497 / 101).
3. Click **Run optimization**. The app fetches daily bars via `ADJUSTED_LAST`, so paper/live permissions need to allow historical data.

## Using CSV data
1. Place your CSV files in the `sample_data` folder.
2. Filenames should match the ticker symbol (e.g., `SPY.csv`, `AAPL.csv`).
3. CSVs must have a date index and columns for `Adj Close` (or `Close`), `High`, and `Low`.
4. In the sidebar, choose `csv` as the data source.

## Controls
- **Universe**: choose tickers (defaults to a diversified S&P 500 subset). Add SPY to ensure regime detection.
- **Lookback / Holding**: ranges and grid step define the optimization surface.
- **Top N / Bottom N**: number of long and optional short legs.
- **Regime settings**: choppiness threshold (lower = more trending) and lookback window for the choppiness calculation.

## Outputs
- **Regime chart**: choppiness index with threshold shading.
- **Heatmaps**: CAGR by lookback/holding for trending and consolidating periods.
- **Best parameter tiles**: CAGR, Sharpe, drawdown, and win rate for each regime.
- **Equity curves**: stepwise cumulative returns for the best trending and consolidating parameter sets.

## Notes
- Yahoo Finance is the fastest way to try the UI; IBKR is available for production-grade data.
- Grid sizes grow quickly; narrow ranges or raise the grid step for faster runs.
- The backtest is simplified (equal-weight positions, no costs). Extend `backtester.py` for slippage, borrow, or sector rules.

import yfinance as yf
import pandas as pd

tickers = ["AAPL", "SPY"]
start = "2023-01-01"
end = "2023-12-31"

print("Downloading...")
data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
print("Columns:", data.columns)
print("Head:\n", data.head())

print("-" * 20)
single_ticker = ["AAPL"]
data_single = yf.download(single_ticker, start=start, end=end, auto_adjust=False, progress=False)
print("Single Ticker Columns:", data_single.columns)
print("Single Ticker Head:\n", data_single.head())

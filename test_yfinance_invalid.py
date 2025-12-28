import yfinance as yf
import pandas as pd

tickers = ["INVALID_TICKER_XYZ"]
start = "2023-01-01"
end = "2023-12-31"

print("Downloading invalid...")
data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
print("Columns:", data.columns)
print("Head:\n", data.head())
print("Empty?", data.empty)

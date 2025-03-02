import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, start="2015-01-01", end="2024-01-01"):
    """Fetch historical stock data from Yahoo Finance."""
    stock_data = yf.download(symbol, start=start, end=end)
    return stock_data

if __name__ == "__main__":
    symbol = "AAPL"  # Example: Apple Inc.
    data = fetch_stock_data(symbol)
    print(data.head())  # Show first 5 rows

import yfinance as yf

ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2025-03-02", progress=False)
print(f"Data fetched: {len(data)} rows")
print(data.head())
print(data.tail())


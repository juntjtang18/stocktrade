import yfinance as yf
import requests

ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2025-03-02"

print("Testing network connection to Yahoo Finance...")
response = requests.get("https://finance.yahoo.com/")
print(f"HTTP Status: {response.status_code} (200 = OK)")

print(f"Fetching {ticker} data from {start_date} to {end_date}...")
data = yf.download(ticker, start=start_date, end=end_date, progress=True)
if data.empty:
    print("Data fetch failed! DataFrame is empty.")
else:
    print(f"Data fetched successfully: {len(data)} rows")
    print("Head:\n", data.head())
    print("Tail:\n", data.tail())


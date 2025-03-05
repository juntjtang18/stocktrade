import yfinance as yf
import pandas as pd
from datetime import datetime

# Parameters
stock_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2025-03-03"  # Current date as per your input

# Function to test yfinance download
def test_yfinance_download(symbol, start, end):
    print(f"Attempting to fetch {symbol} data from {start} to {end}...")
    try:
        # Fetch data with verbose output
        data = yf.download(symbol, start=start, end=end, progress=True)
        if data.empty:
            print(f"Error: Downloaded data for {symbol} is empty.")
            return None
        print(f"Successfully fetched {len(data)} rows of data.")
        print(f"Sample data:\n{data.tail(5)}")
        return data
    except Exception as e:
        print(f"Error encountered: {str(e)}")
        return None

# Run the test
data = test_yfinance_download(stock_symbol, start_date, end_date)

# Additional diagnostics if data fetch fails
if data is None:
    print("\nRunning additional diagnostics...")
    try:
        # Test with a smaller date range
        print("Trying a shorter range (last 7 days)...")
        short_data = yf.download(stock_symbol, period="7d", progress=True)
        if not short_data.empty:
            print(f"Short range worked! Fetched {len(short_data)} rows.")
            print(f"Sample data:\n{short_data.tail(5)}")
        else:
            print("Short range also failed.")
    except Exception as e:
        print(f"Short range test failed: {str(e)}")

    # Test connectivity with a simple ticker info fetch
    try:
        print("\nTesting basic ticker info fetch...")
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info
        print(f"Ticker info fetched successfully: {info['longName']}")
    except Exception as e:
        print(f"Ticker info fetch failed: {str(e)}")

print("\nDiagnostics complete.")
# .\mylib\download_data.py
import pandas as pd
import requests
import os
import yfinance as yf
from datetime import date, timedelta

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def download_data(stock_symbol, api_key="nymCOaghA4Yf08LppHIKqpIagI5Z6Vy0", source="polygon"):
    """
    Download stock data from Polygon.io, yfinance, or Alpha Vantage and save to CSV in a uniform format.
    
    Parameters:
    - stock_symbol (str): Stock ticker symbol (e.g., "AAPL")
    - api_key (str): API key for Polygon.io or Alpha Vantage
    - source (str): Data source ("polygon", "yfinance", "alpha")
    
    Returns:
    - pd.DataFrame: Stock data with Date (index), Open, High, Low, Close, Volume
    """
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    local_file = os.path.join(data_dir, f"{stock_symbol}_data.csv")
    today = date.today()
    end_date = today.strftime("%Y-%m-%d")
    start_date = "2020-01-01"

    def is_data_up_to_date(file_path):
        if not os.path.exists(file_path):
            return False
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True, nrows=1)
            latest_date = data.index[0].date()
            threshold_date = today - timedelta(days=1)
            return latest_date >= threshold_date
        except Exception:
            return False

    if not is_data_up_to_date(local_file):
        print(f"Data for {stock_symbol} is missing or outdated. Starting download from {source}...")
        
        if source == "polygon":
            url = f"https://api.polygon.io/v2/aggs/ticker/{stock_symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=desc&limit=50000&apiKey={api_key}"
            response = requests.get(url).json()
            if response.get("status") not in ["OK", "DELAYED"] or "results" not in response:
                raise ValueError(f"Polygon.io fetch failed: {response.get('error', 'Unknown error')}")
            data = pd.DataFrame(response["results"])
            data["Date"] = pd.to_datetime(data["t"], unit="ms").dt.date  # Strip time
            data = data.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

        elif source == "yfinance":
            data = yf.download(stock_symbol, start=start_date, end=end_date)
            data = data.reset_index()  # Move Date from index to column
            data["Date"] = data["Date"].dt.date  # Ensure no time component
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]  # Drop Adj Close

        elif source == "alpha":
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={api_key}"
            response = requests.get(url).json()
            if "Time Series (Daily)" not in response:
                raise ValueError(f"Alpha Vantage fetch failed: {response.get('Information', 'Unknown error')}")
            data = pd.DataFrame(response["Time Series (Daily)"]).T
            data = data.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
            data.index = pd.to_datetime(data.index).date  # Convert index to date only
            data = data.reset_index().rename(columns={"index": "Date"})
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        
        else:
            raise ValueError(f"Unknown source: {source}")

        # Convert to uniform format
        data["Date"] = pd.to_datetime(data["Date"]).dt.date  # Ensure date only
        data.set_index("Date", inplace=True)
        data = data[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        data.sort_index(ascending=False, inplace=True)  # Descending order
        data.to_csv(local_file)
        print(f"Download complete. Data saved to {local_file} from {source}")
    else:
        print(f"Data for {stock_symbol} is already up-to-date. Skipping download and using local file: {local_file}")

    data = pd.read_csv(local_file, index_col=0, parse_dates=True)
    return data

if __name__ == "__main__":
    data = download_data("AAPL", source="polygon")  # Switch to "yfinance" or "alpha" as needed
    print(data.head())
# .\mylib\download_data.py
import pandas as pd
import requests
import os
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import holidays  # New import for holiday handling

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
    today = datetime.today().date()
    end_date = today.strftime("%Y-%m-%d")
    start_date = "2020-01-01"

    def get_last_trading_date(current_date_time):
        """Return the last trading date, accounting for weekends and U.S. holidays."""
        et_tz = pytz.timezone('America/New_York')
        current_date = current_date_time.date()
        weekday = current_date.weekday()  # 0 = Monday, 6 = Sunday
        
        # Initialize U.S. holidays (NYSE typically follows these)
        us_holidays = holidays.US(years=[current_date.year, current_date.year - 1])
        
        # Check if before or after 4:30 PM ET
        closing_time = current_date_time.replace(hour=16, minute=30, second=0, microsecond=0, tzinfo=et_tz)
        
        if current_date_time < closing_time:  # Before 4:30 PM ET
            if weekday == 0:  # Monday
                candidate_date = current_date - timedelta(days=3)  # Previous Friday
            else:
                candidate_date = current_date - timedelta(days=1)  # Previous day
        else:  # After 4:30 PM ET
            if weekday == 5:  # Saturday
                candidate_date = current_date - timedelta(days=1)  # Friday
            elif weekday == 6:  # Sunday
                candidate_date = current_date - timedelta(days=2)  # Friday
            else:
                candidate_date = current_date  # Today (Monday-Friday)

        # Step back if candidate_date is a weekend or holiday
        while candidate_date.weekday() >= 5 or candidate_date in us_holidays:
            candidate_date -= timedelta(days=1)
        
        return candidate_date

    def is_data_up_to_date(file_path):
        if not os.path.exists(file_path):
            return False
        try:
            # Get current time in Eastern Time
            et_tz = pytz.timezone('America/New_York')
            current_date_time = datetime.now(et_tz)
            
            # Read the latest date from the file
            data = pd.read_csv(file_path, index_col=0, parse_dates=True, nrows=1)
            file_date = data.index[0].date()
            
            # Get the last trading date
            last_trading_date = get_last_trading_date(current_date_time)
            
            # If last trading date is newer than file date, download
            return last_trading_date <= file_date
        except Exception as e:
            print(f"Error checking data freshness: {e}")
            return False

    if not is_data_up_to_date(local_file):
        print(f"Data for {stock_symbol} is missing or outdated. Starting download from {source}...")
        
        if source == "polygon":
            url = f"https://api.polygon.io/v2/aggs/ticker/{stock_symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=desc&limit=50000&apiKey={api_key}"
            response = requests.get(url).json()
            if response.get("status") not in ["OK", "DELAYED"] or "results" not in response:
                raise ValueError(f"Polygon.io fetch failed: {response.get('error', 'Unknown error')}")
            data = pd.DataFrame(response["results"])
            data["Date"] = pd.to_datetime(data["t"], unit="ms").dt.date
            data = data.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

        elif source == "yfinance":
            data = yf.download(stock_symbol, start=start_date, end=end_date)
            data = data.reset_index()
            data["Date"] = data["Date"].dt.date
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

        elif source == "alpha":
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={api_key}"
            response = requests.get(url).json()
            if "Time Series (Daily)" not in response:
                raise ValueError(f"Alpha Vantage fetch failed: {response.get('Information', 'Unknown error')}")
            data = pd.DataFrame(response["Time Series (Daily)"]).T
            data = data.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
            data.index = pd.to_datetime(data.index).date
            data = data.reset_index().rename(columns={"index": "Date"})
            data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        
        else:
            raise ValueError(f"Unknown source: {source}")

        # Convert to uniform format
        data["Date"] = pd.to_datetime(data["Date"]).dt.date
        data.set_index("Date", inplace=True)
        data = data[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        data.sort_index(ascending=False, inplace=True)
        data.to_csv(local_file)
        print(f"Download complete. Data saved to {local_file} from {source}")
    else:
        print(f"Data for {stock_symbol} is already up-to-date. Skipping download and using local file: {local_file}")

    data = pd.read_csv(local_file, index_col=0, parse_dates=True)
    return data

if __name__ == "__main__":
    data = download_data("AAPL", source="polygon")
    print(data.head())
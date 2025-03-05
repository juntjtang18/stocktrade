# test_polygon.py
import requests
import pandas as pd

api_key = "nymCOaghA4Yf08LppHIKqpIagI5Z6Vy0"
stock_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2025-03-04"

url = f"https://api.polygon.io/v2/aggs/ticker/{stock_symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=desc&limit=50000&apiKey={api_key}"
response = requests.get(url).json()

print("API Response:", response)

# Check if the request was successful (accept 'OK' or 'DELAYED')
if response.get("status") in ["OK", "DELAYED"] and "results" in response:
    # Process the data
    data = pd.DataFrame(response["results"])
    data["t"] = pd.to_datetime(data["t"], unit="ms")
    data = data.rename(columns={
        "t": "Date",
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume"
    })
    data.set_index("Date", inplace=True)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data = data.astype(float)
    
    # Check if we got the full date range
    earliest_date = data.index.min().strftime("%Y-%m-%d")
    if earliest_date > start_date:
        print(f"Warning: Only partial data retrieved. Earliest date: {earliest_date}, requested: {start_date}")
    
    data.to_csv("AAPL_test.csv")
    print("Data saved to AAPL_test.csv")
    print(data.head())
else:
    error_msg = response.get("error", "Unknown error from Polygon.io")
    print(f"Failed to retrieve data: {error_msg}")
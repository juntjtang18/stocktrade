



import pandas as pd
import requests

api_key2 = "U05ZXJI8JWTSYBNP"  # Get from Alpha Vantage
api_key  = "WTW6JPZU827ZHUTH"
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=full&apikey={api_key}"
response = requests.get(url).json()
print(response)
data = pd.DataFrame(response["Time Series (Daily)"]).T
data.to_csv("AAPL_data.csv")
print("Saved to AAPL_data.csv")
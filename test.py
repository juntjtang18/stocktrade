import requests
response = requests.get("https://finance.yahoo.com/", headers={'User-Agent': 'Mozilla/5.0'})
print(f"HTTP Status: {response.status_code}")


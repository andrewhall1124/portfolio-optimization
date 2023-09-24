import requests
import pandas as pd

# Call twelvedata api for all tickers
url = "https://api.twelvedata.com/stocks"
params = {
  "country": "United States"
}
response = requests.get(url, params)

# Parse data
data = response.json()
data = pd.DataFrame(data['data'])
data = data.reindex(columns=['symbol', 'name', ])

# Write to csv
data.to_csv('all-tickers.csv', index=False)

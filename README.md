# Portfolio Optimization

# Setup
This python script uses the TwelveData API
- Get your own API key at https://twelvedata.com/
- Create a config.ini file
- It should look like this (do not surround the key in quotation marks)

[API]

API_KEY = your-api-key

- The batch-portfolio.py file utilizes a batch API call to improve speed
- Note that the TwelveData API doesn't allow for more than 8 API calls per minute
- Make sure to match the number of tickers with the number of weights
- The weights should also add to 1


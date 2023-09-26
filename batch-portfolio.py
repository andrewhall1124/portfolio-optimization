import requests as rq
import pandas as pd
import numpy as np
import math
import time
import configparser
import sys

config = configparser.ConfigParser()
config.read('config.ini')

# Record the start time
start_time = time.time()

tickers = ['ibm','aapl','meta']
weights = np.array([.3, .3, .4])
start_date = '2020-09-23 15:03:00'
end_date = '2023-09-23 15:04:00'
interval = '1week'

# Check for matching lengths of tickers and weights
if len(tickers) != len(weights):
  print(f"\nERROR: Tickers and weights length do not match. Tickers has length {len(tickers)} and weights has length {len(weights)}\n")
  sys.exit(0)

# Fetching stock data from 12 Data API
API_KEY = config['API']['API_KEY']
url = f"https://api.twelvedata.com/time_series?symbol={','.join(tickers)}&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={API_KEY}"
print(f"\nFetching stock data for \"{','.join(tickers)}\"")
response = rq.get(url)
stock_data_list = []
if response.status_code == 200:
  for ticker in tickers:
    print(f"Parsing \"{ticker}\"")
    data = response.json()
    data = data[f"{ticker}"]['values']
    data = pd.DataFrame(data)
    data = data.sort_values(by="datetime", ascending=True)
    data = data.reindex(columns=['datetime', 'close'])
    data['close'] = pd.to_numeric(data['close'])
    data['return'] = data['close'].pct_change() * 100
    data['return'] = data['return'].fillna(0)
    average_returns = data['return'].mean()
    std_return = data['return'].std()
    returns = [float(x) for x in data['return'].values]
    stock_data = {
        "ticker": ticker,
        "average_return": average_returns,
        "std_return": std_return,
        "returns": returns
    }
    stock_data_list.append(stock_data)
    print(f"Added \"{ticker}\" to stock data list")
else:
  print(f"Request failed with status code: {response.status_code}") 

# Calculate covariance matrix
print("Calculating covariance matrix\n")
stock_returns_matrix = np.array([stock['returns'] for stock in stock_data_list])
average_returns = np.mean(stock_returns_matrix, axis = 1)
covariance_matrix = np.cov(stock_returns_matrix)

print("Stock returns matrix  ", stock_returns_matrix)
print("Average returns array  ", average_returns)
print("Covariance matrix  ", covariance_matrix)

    
# Function for calculating portfolio return
def calculate_portfolio_return(weights, stock_data_list):
  stock_returns = np.array([stock['average_return'] for stock in stock_data_list])
  return np.dot(weights,stock_returns)

# Function for calculating portfolio variance
def calculate_portfolio_std(weights, covariance_matrix):
  return math.sqrt(np.dot(weights,np.matmul(covariance_matrix,weights)))

# Function for calculating the sharpe ratio
def calculate_portfolio_sharpe(weights, covariance_matrix, stock_data_list):
  portfolio_return = calculate_portfolio_return(weights,stock_data_list)
  portfolio_std = calculate_portfolio_std(weights, covariance_matrix)
  return portfolio_return / portfolio_std

print(f"Tickers: {tickers}")
print(f"Weights: {weights}")
print(f"Portfolio return: {calculate_portfolio_return(weights,stock_data_list)}")
print(f"Portfolio variance: {calculate_portfolio_std(weights,covariance_matrix)}")
print(f"Portfolio sharpe: {calculate_portfolio_sharpe(weights, covariance_matrix, stock_data_list)}")

# Output elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nExecuted in {elapsed_time} seconds\n")
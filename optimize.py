import pandas as pd
import numpy as np
from scipy import optimize
import requests as rq
import sys
import configparser
import time

config = configparser.ConfigParser()
config.read('config.ini')

# Record the start time
start_time = time.time()

# tickers = ['ibm','aapl','meta','googl','adbe', 'dis', 'adsk', 'nvda']
tickers = ['ibm','aapl','meta', 'googl']
weights = np.array([.25,.25,.25,.25])
start_date = '2020-09-23 15:03:00'
end_date = '2023-09-23 15:04:00'
interval = '1week'
portfolio_size = len(tickers)
risk_free = 5.46
risk_free = (np.power((1 + risk_free),  (1.0 / 52.0)) - 1.0) * 100 

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
    average_return = data['return'].mean()
    std_return = data['return'].std()
    returns = [float(x) for x in data['return'].values]
    stock_data = {
        "ticker": ticker,
        "average_return": average_return,
        "std_return": std_return,
        "returns": returns
    }
    stock_data_list.append(stock_data)
    print(f"Added \"{ticker}\" to stock data list")
else:
  print(f"Request failed with status code: {response.status_code}") 

# Compute average_returns
average_returns = np.array([stock['average_return'] for stock in stock_data_list])

# Calculate covariance matrix
print("Calculating covariance matrix\n")
returns_array = np.array([stock['returns'] for stock in stock_data_list])
covariance_matrix = np.cov(returns_array)

# Function for calculating portfolio return
def calculate_portfolio_return(weights, average_returns):
  return np.dot(weights,average_returns)

# Function for calculating portfolio variance
def calculate_portfolio_std(weights, covariance_matrix):
  return np.sqrt(np.dot(np.matmul(weights, covariance_matrix), weights))

def optimize_sharpe(average_returns, covariance_matrix, risk_free, portfolio_size):
    
    # define maximization of Sharpe Ratio
    def  f(weights, average_returns, covariance_matrix, risk_free):
        funcDenomr = calculate_portfolio_std(weights, covariance_matrix)
        funcNumer = calculate_portfolio_return(weights, average_returns)-risk_free
        func = -(funcNumer / funcDenomr)
        return func

    #define equality constraint representing fully invested portfolio
    def constraintEq(x):
        A=np.ones(x.shape)
        b=1
        constraintVal = np.matmul(A,x.T)-b 
        return constraintVal
    
    
    #define bounds and other parameters
    xinit=np.repeat(0.33, portfolio_size)
    cons = ({'type': 'eq', 'fun':constraintEq})
    lb = 0
    ub = 1
    bnds = tuple([(lb,ub) for x in xinit])
    
    #invoke minimize solver
    opt = optimize.minimize (
      f, x0 = xinit, args = (average_returns, covariance_matrix,\
      risk_free, portfolio_size), method = 'SLSQP',  \
      bounds = bnds, constraints = cons, tol = 10**-3
    )
    
    return opt

print(f"Tickers: {tickers}")
print(f"Weights: {weights}")
print(f"Portfolio return: {calculate_portfolio_return(weights, average_returns)}")
print(f"Portfolio variance: {calculate_portfolio_std(weights,covariance_matrix)}")

# Output elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nExecuted in {elapsed_time} seconds\n")
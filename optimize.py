import numpy as np
import pandas as pd
from scipy import optimize 
import configparser
import time
import requests as rq

config = configparser.ConfigParser()
config.read('config.ini')

# Record the start time
start_time = time.time()

tickers = ['f', 'dis', 'vz', 'nvda']
start_date = '2020-09-23 15:03:00'
end_date = '2023-09-23 15:04:00'
interval = '1day'

## Fetching and setting up all of the data ##

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
    stock_returns = [float(x) for x in data['return'].values]
    stock_data_list.append(stock_returns)
    print(f"Added \"{ticker}\" to stock data list")
else:
  print(f"Request failed with status code: {response.status_code}") 

# Calculate covariance matrix
print("Calculating covariance_matrix and average_returns array\n")
stock_returns_matrix = np.array(stock_data_list)
average_returns = np.mean(stock_returns_matrix, axis = 1)
covariance_matrix = np.cov(stock_returns_matrix)

## Optimization function ##

def maximize_that_fricking_sharpe_ratio(average_returns, covariance_matrix, risk_free, portfolio_size):
    
    # define maximization of Sharpe Ratio using principle of duality
    def  f(x, average_returns, covariance_matrix, risk_free, portfolio_size):
        funcDenomr = np.sqrt(np.matmul(np.matmul(x, covariance_matrix), x.T) )
        funcNumer = np.matmul(np.array(average_returns),x.T)-risk_free
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
    opt = optimize.minimize (f, x0 = xinit, args = (average_returns, covariance_matrix,\
                             risk_free, portfolio_size), method = 'SLSQP',  \
                             bounds = bnds, constraints = cons, tol = 10**-3)
    
    return opt


## Now the fun begins ##

#Initialize portfolio
portfolio_size = len(tickers)
annual_risk_free= 5.4 / 100
daily_risk_free = (np.power((1 + annual_risk_free),  (1.0 / 360.0)) - 1.0) * 100 

#Compute maximal Sharpe Ratio and optimal weights
result = maximize_that_fricking_sharpe_ratio(average_returns, covariance_matrix, daily_risk_free, portfolio_size)
optimal_weights = np.array([result.x])

#Compute other metrics  
portfolio_risk = np.matmul((np.matmul(optimal_weights,covariance_matrix)), np.transpose(optimal_weights))
portfolio_return = np.matmul(np.array(average_returns),optimal_weights.T)
annualized_risk = np.sqrt(portfolio_risk*251) 
annualize_return = 251*np.array(portfolio_return) 
sharpe_ratio = (annualize_return-annual_risk_free)/annualized_risk 

# Set precision for printing results
np.set_printoptions(precision=3, suppress = True)

# Display results
print('Maximal Sharpe Ratio:  ', sharpe_ratio)
print('\nAnnualized Risk (%):  ', annualized_risk)
print('\nAnnualized Expected Portfolio Return (%):  ', annualize_return)
print('\nOptimal weights (%):')

# Print the optimal weights with the ticker symbol
for i in range(len(tickers)):
  print('{} | {:.2f}'.format(tickers[i], optimal_weights[0][i] * 100))

# Output elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nExecuted in {elapsed_time} seconds\n")
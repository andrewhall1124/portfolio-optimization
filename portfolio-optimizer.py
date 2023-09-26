import requests as rq
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math
import time
import configparser
import sys

config = configparser.ConfigParser()
config.read("config.ini")

# Record the start time
start_time = time.time()

tickers = ["aapl", "ibm", "meta", "googl", "adbe"]
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
start_date = "2020-09-23 15:03:00"
end_date = "2023-09-23 15:04:00"
interval = "1week"

# Check for matching lengths of tickers and weights
if len(tickers) != len(weights):
    print(
        f"\nERROR: Tickers and weights length do not match. Tickers has length {len(tickers)} and weights has length {len(weights)}\n"
    )
    sys.exit(0)


# Function for fetching stock data from 12 Data API
def fetch_stock_data(symbol):
    API_KEY = config["API"]["API_KEY"]
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={API_KEY}"
    response = rq.get(url)
    print(f'Fetching stock data for "{symbol}"')

    if response.status_code == 200:
        data = response.json()["values"]
        data = pd.DataFrame(data)
        data = data.sort_values(by="datetime", ascending=True)
        data = data.reindex(columns=["datetime", "close"])
        data["close"] = pd.to_numeric(data["close"])
        data["return"] = data["close"].pct_change() * 100
        data["return"] = data["return"].fillna(0)

        average_return = data["return"].mean()
        std_return = data["return"].std()
        returns = [float(x) for x in data["return"].values]

        return {
            "average_return": average_return,
            "std_return": std_return,
            "returns": returns,
        }

    else:
        print(f"Request failed with status code: {response.status_code}")


# Loop for fetching stock data for each ticker
print("\nCreating stock data list")
stock_data_list = []
for stock in tickers:
    stock_data = fetch_stock_data(stock)
    stock_data_list.append(stock_data)
    print(f'Adding "{stock}" to stock data list')

# Calculate covariance matrix
print("Calculating covariance matrix\n")
covariance_matrix = []
for stock1 in stock_data_list:
    covariance_row = []
    for stock2 in stock_data_list:
        arr1 = np.array(stock1["returns"])
        arr2 = np.array(stock2["returns"])
        covariance_value = (np.cov(arr1, arr2))[0, 1]
        covariance_row.append(covariance_value)
    covariance_matrix.append(covariance_row)
covariance_matrix = np.array(covariance_matrix)


# Function for calculating portfolio return
def calculate_portfolio_return(weights, stock_data_list):
    stock_returns = np.array([stock["average_return"] for stock in stock_data_list])
    return np.dot(weights, stock_returns)


# Function for calculating portfolio variance
def calculate_portfolio_std(weights, covariance_matrix):
    return math.sqrt(np.dot(weights, np.matmul(covariance_matrix, weights)))


# Function for calculating the sharpe ratio
def calculate_portfolio_sharpe(weights, covariance_matrix, stock_data_list):
    portfolio_return = calculate_portfolio_return(weights, stock_data_list)
    portfolio_std = calculate_portfolio_std(weights, covariance_matrix)
    return portfolio_return / portfolio_std


print(f"Tickers: {tickers}")
print(f"Weights: {weights}")
print(f"Portfolio return: {calculate_portfolio_return(weights,stock_data_list)}")
print(f"Portfolio variance: {calculate_portfolio_std(weights,covariance_matrix)}")
print(
    f"Portfolio sharpe: {calculate_portfolio_sharpe(weights, covariance_matrix, stock_data_list)}"
)

print("Solving")
sol = minimize(
    lambda w: -calculate_portfolio_sharpe(
        w / np.sum(w), covariance_matrix, stock_data_list
    ),
    weights,
    bounds=((0.0, 1.0),) * len(weights),
    constraints=({"type": "eq", "fun": lambda w: 1 - sum(w)}),
)

print("Best weights", sol.x / np.sum(sol.x))
print(
    f"Optimized sharpe: {calculate_portfolio_sharpe(sol.x / np.sum(sol.x), covariance_matrix, stock_data_list)}"
)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"\nExecuted in {elapsed_time} seconds\n")

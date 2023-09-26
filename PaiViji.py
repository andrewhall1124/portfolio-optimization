import numpy as np
import pandas as pd
from scipy import optimize 

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

## Fetching and setting up all of the data ##

#Calculate average_returns and covariance_matrix
stock_file_name = 'DJIA_Apr112014_Apr112019.csv'
m = 1259  #excluding header
columns = 29  #excluding date

#Read in stock prices 
df = pd.read_csv(stock_file_name,  nrows= m)

#Extract ticker symbols
ticker_symbols = df.columns[1:columns+1].tolist()

#Remove timedate from stock data
stock_data_df = df.iloc[0:, 1:]

#Calculate stock returns matrix
stock_prices_matrix = np.asarray(stock_data_df)
[m, n]=stock_prices_matrix.shape

def compute_stock_returns(stock_price, rows, columns):    
    stock_return = np.zeros([rows-1, columns])
    for j in range(columns):        # j: Stocks
        for i in range(rows-1):     # i: Daily Prices
            stock_return[i,j]=((stock_price[i+1, j]-stock_price[i,j])/stock_price[i,j])* 100

    return stock_return

stock_returns_matrix = compute_stock_returns(stock_prices_matrix, m, n)

#Calculate average_returns and covariance matrix
average_returns = np.mean(stock_returns_matrix, axis = 0)
covariance_matrix = np.cov(stock_returns_matrix, rowvar=False)

## Now the fun begins ##

#Initialize portfolio
portfolio_size = columns
annual_risk_free= 5.4 / 100
daily_risk_free = (np.power((1 + annual_risk_free),  (1.0 / 360.0)) - 1.0) * 100 

minRiskPoint = []
expPortfolioReturnPoint =[]
sharpe_ratio = 0

#Compute maximal Sharpe Ratio and optimal weights
result = maximize_that_fricking_sharpe_ratio(average_returns, covariance_matrix, daily_risk_free, portfolio_size)
optimal_weights = np.array([result.x])

#Compute other metrics  
portfolio_risk = np.matmul((np.matmul(optimal_weights,covariance_matrix)), np.transpose(optimal_weights))
portfolio_return = np.matmul(np.array(average_returns),optimal_weights.T)
annualized_risk = np.sqrt(portfolio_risk*251) 
annualize_return = 251*np.array(portfolio_return) 
sharpe_ratio = (annualize_return-annual_risk_free)/annualized_risk 

#Set precision for printing results
np.set_printoptions(precision=3, suppress = True)

#Display results
print('Maximal Sharpe Ratio: ', sharpe_ratio)
print('\nAnnualized Risk (%):  ',annualized_risk)
print( '\nAnnualized Expected Portfolio Return(%):  ', annualize_return)
print('\nOptimal weights (%):\n',  optimal_weights.T*100 )
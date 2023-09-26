import numpy as np
import pandas as pd
from scipy import optimize 

#function to undertake Sharpe Ratio maximization subject to basic constraints of the portfolio

# Portfolio size should be 29 from the dataset

def MaximizeSharpeRatioOptmzn(average_returns, covariance_matrix, risk_free, portfolio_size):
    
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






#Calculate average_returns and covariance_matrix
#input k portfolio 1 dataset comprising 15 stocks
stock_file_name = 'DJIA_Apr112014_Apr112019.csv'
m = 1259  #excluding header
columns = 29  #excluding date

#read stock prices 
df = pd.read_csv(stock_file_name,  nrows= m)
print(df)

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


#set precision for printing results
# np.set_printoptions(precision=10, suppress = True)

#Calculate average_returns and covariance matrix
average_returns = np.mean(stock_returns_matrix, axis = 0)
covariance_matrix = np.cov(stock_returns_matrix, rowvar=False)






#obtain maximal Sharpe Ratio for k-portfolio 1 of Dow stocks

#set portfolio size
portfolio_size = columns

#set risk free asset rate of return
Rf=3  # April 2019 average risk  free rate of return in USA approx 3%
annRiskFreeRate = Rf/100

#compute daily risk free rate in percentage
r0 = (np.power((1 + annRiskFreeRate),  (1.0 / 360.0)) - 1.0) * 100 
print('\nRisk free rate (daily %): ', end="")
print ("{0:.3f}".format(r0)) 

#initialization
xOptimal =[]
minRiskPoint = []
expPortfolioReturnPoint =[]
maxSharpeRatio = 0

#compute maximal Sharpe Ratio and optimal weights
result = MaximizeSharpeRatioOptmzn(average_returns, covariance_matrix, r0, portfolio_size)
xOptimal.append(result.x)

    
#compute risk returns and max Sharpe Ratio of the optimal portfolio   
xOptimalArray = np.array(xOptimal)
Risk = np.matmul((np.matmul(xOptimalArray,covariance_matrix)), np.transpose(xOptimalArray))
expReturn = np.matmul(np.array(average_returns),xOptimalArray.T)
annRisk =   np.sqrt(Risk*251) 
annRet = 251*np.array(expReturn) 
maxSharpeRatio = (annRet-Rf)/annRisk 

#set precision for printing results
np.set_printoptions(precision=3, suppress = True)


#display results
print('Maximal Sharpe Ratio: ', maxSharpeRatio, '\nAnnualized Risk (%):  ', \
      annRisk, '\nAnnualized Expected Portfolio Return(%):  ', annRet)
print('\nOptimal weights (%):\n',  xOptimalArray.T*100 )
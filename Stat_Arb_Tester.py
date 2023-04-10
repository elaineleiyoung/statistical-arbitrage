

import numpy as np #importing numpy: library for scientific computing in Python
import pandas as pd #importing pandas: library for data manipulation/analysis (dataframe is key)
import matplotlib.pyplot as plt #importing matplotlib: library for data visualization
import statsmodels.tsa.stattools as ts #importing statistical tools for time series analysis (for Aug Dickey-Fuller Test)
from scipy.stats import linregress #import linear regression code
import yfinance as yf #importing Yahoo Finance database
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import datetime #importing date/time data

#GOAL: COMPARE 10 FINANCIAL SERVICES STOCKS, FIND THE OPTIMAL COINTEGRATED PAIR AND DO PAIRS-TESTING ON IT
tickers = ['V', 'JPM', 'MA', 'BAC', 'C', 'MS', 'HSBC', 'WFC', 'AXP', 'GS']
#Visa, JP Morgan, Mastercard, BofA, CitiGroup, Morgan Stanley, HSBC, Wells Fargo, American Express, Goldman Sachs

#yfinance data download for stock prices
data = yf.download(tickers, start="2019-01-01", end="2023-02-22")
startdate = datetime.datetime(2019, 1, 1)
todate = datetime.datetime(2023, 2, 22)

#data = pd.DataFrame()

compare = 'MA'
#crypto?

#pearson correlation coefficient, only those above 0.8

#augmented dickey-fuller test to determine cointegration, only valid if p<0.05
def adf(series):
    return ts.adfuller(series)[1]

#zscore to determine spread
def zscore(series):
    return (series -series.mean())/np.std(series)

success = []

#comparing all stocks' correlation/cointegration to JPM:
for i in range(0,10):
    #as long as ticker isn't equal to JPM,
    if compare == tickers[i]:
        continue

    tickername = tickers[i]
    #calculate ratio between two
    ratio = data['Adj Close'][compare]/data['Adj Close'][tickername]
    #calculate p-score
    p = adf(ratio)
    #calculate r score
    r = np.corrcoef(data['Adj Close'][compare], data['Adj Close'][tickername])

    print("Ticker MA vs {}, p is {}, r is {}".format(tickername, p, r))
    #good arbitrage pair
    if p < 0.05 or r.any() > 0.80:
        np.append(success, tickername)

print(success)
#printing correlation matrix to get basic idea of relationship
fig, ax1 = plt.subplots(figsize=(10,7))
sns.heatmap(data['Adj Close'].pct_change().corr(method='pearson'), ax=ax1, cmap='coolwarm', annot=True, fmt=".2f")
ax1.set_title('Assets Correlation Matrix')
plt.show()

#print p-value matrix? 
def coint_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n,n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            result = coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i, j] = result[1]
            if result[1] < 0.05:
                pairs.append((keys[i], keys[j]))
    return pvalue_matrix, pairs 

model = sm.OLS(data['Adj Close']['MA'], data['Adj Close']['V']).fit()
#asset 1 is JPM, asset 2 is MA

plt.rc('figure', figsize=(12,7))
plt.text(0.20, 0.30, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(left=-0.1, right=0.8, top=0.7, bottom=0.1)
plt.show()
#plt.savefig('images/chart4', dpi=300)

pvalues, pairs = coint_pairs(data['Adj Close'])
fig, ax2 = plt.subplots(figsize=(10,7))
sns.heatmap(pvalues, xticklabels = data['Adj Close'].columns, yticklabels = data['Adj Close'].columns, cmap = 'RdYlGn_r', annot = True, fmt = ".2f", mask = (pvalues >= 0.99))
ax2.set_title('Assets Cointregation Matrix p-values Between Pairs')
plt.tight_layout()
plt.show()

hedger = model.params[0]
spread = data['Adj Close']['MA'] - (hedger*data['Adj Close']['V'])
plt.plot(spread)
plt.xlabel('Date')
plt.ylabel('Spread')
plt.grid(True)
#ax = spread.plot(figsize=(12,6), title = "Pair's Spread")
#ax.set_ylabel("Spread")
#ax.grid(True)

print(hedger)

#test graphs to ensure hedging ratio normalizes graphs
plt.plot(np.log(data['Adj Close']['MA']))
plt.plot(np.log(data['Adj Close']['V'])*hedger)
plt.xlabel('Date')
plt.ylabel('Price Normalized')
plt.show()
#spread = np.log(data['Adj Close']['JPM']) - (hedger * np.log(data['Adj Close']['MA']))

#we find that JPM and MA is most cointegrated (though not <0.05), also highly correlated
plt.plot(np.log(data['Adj Close']['MA'])-0.90)
plt.plot(np.log(data['Adj Close']['V']))
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
#how to normalize data? 
# compare my p-values to code's p-values



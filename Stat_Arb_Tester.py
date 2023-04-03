

import numpy as np #importing numpy: library for scientific computing in Python
import pandas as pd #importing pandas: library for data manipulation/analysis (dataframe is key)
import matplotlib.pyplot as plt #importing matplotlib: library for data visualization
import statsmodels.tsa.stattools as ts #importing statistical tools for time series analysis (for Aug Dickey-Fuller Test)
from scipy.stats import linregress #import linear regression code
import yfinance as yf #importing Yahoo Finance database
import seaborn as sns
import datetime #importing date/time data

#GOAL: COMPARE 10 FINANCIAL SERVICES STOCKS, FIND THE OPTIMAL COINTEGRATED PAIR AND DO PAIRS-TESTING ON IT
tickers = ['V', 'JPM', 'MA', 'BAC', 'C', 'MS', 'HSBC', 'WFC', 'AXP', 'GS']
#Visa, JP Morgan, Mastercard, BofA, CitiGroup, Morgan Stanley, HSBC, Wells Fargo, American Express, Goldman Sachs

#yfinance data download for stock prices
data = yf.download(tickers, start="2019-01-01", end="2023-02-22")
startdate = datetime.datetime(2019, 1, 1)
todate = datetime.datetime(2023, 2, 22)

#data = pd.DataFrame()

compare = 'JPM'

#data['V'] = data['Adj Close']['V']
#data['JPM'] = data['Adj Close']['JPM']
#data['MA'] = data['Adj Close']['MA']
#data['BAC'] = data['Adj Close']['BAC']
#data['C'] = data['Adj Close']['C']
#data['MS'] = data['Adj Close']['MS']
#data['HSBC'] = data['Adj Close']['HSBC']
#data['WFC'] = data['Adj Close']['WFC']
#data['AXP'] = data['Adj Close']['AXP']
#data['GS'] = data['Adj Close']['GS']

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

    print("Ticker JPM vs {}, p is {}, r is {}".format(tickername, p, r))
    #good arbitrage pair
    if p < 0.05 or r.any() > 0.80:
        np.append(success, tickername)

print(success)

def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n,n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1,n):
            result = ts.coint(data[keys[i]], data[keys[j]])
            pvalue_matrix[i,j] = result[1]
            if result[1] < 0.05:
                pairs.append((keys[i], keys[j]))
    return pvalue_matrix, pairs

pvalues, pairs = find_cointegrated_pairs(train_close)
print(pairs)
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(pvalues, xticklabels = train_close.columns, yticklabels = train_close.columns, cmap = 'RdYlGn_r', annot = True, fmt=".2f", mask = (pvalues >= 0.99))
ax.set_title('Assets Cointregation Matrix p-values Between Pairs')
plt.tight_layout()
plt.savefig('images/chart2', dpi = 300)     


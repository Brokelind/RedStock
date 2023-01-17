'''import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
'''
import yfinance as yf
import pandas as pd
# Read and print the stock tickers that make up S&P500
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers.head()

# Get the data for this tickers from yahoo finance
ticker_data = yf.download(tickers.Symbol.to_list(),'2021-1-1','2022-12-19', auto_adjust=True)['Close']
#ticker_data.head()


ticker_data.to_csv('input/ticker_data.csv', header = True, encoding="utf-8",
index=True)
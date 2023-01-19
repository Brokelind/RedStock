
import yfinance as yf
import pandas as pd




# Read and print the stock tickers that make up S&P500
stock_list = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = stock_list[0]
tickers.head()

tickers.Symbol.to_csv('input/tickers.csv', header = True, encoding="utf-8",
index=False)

tickers.Security.to_csv('input/tickers_fullname.csv', header = True, encoding="utf-8",
index=False)

# Get the data for this tickers from yahoo finance
ticker_data = yf.download(tickers.Symbol.to_list(),'2021-1-1','2022-12-19', auto_adjust=True)['Close']
#ticker_data.head()


ticker_data.to_csv('input/ticker_data.csv', header = True, encoding="utf-8",
index=True)
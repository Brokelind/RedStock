import csv
from re import A
from numpy import append
from yfinance import ticker
import pandas as pd
import datetime
from sklearn import preprocessing
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor



def XY_mapping(control_period = 30):

    #list of stock tickers
    tickers = []
    with open('input/tickers.csv', newline = '') as csvtick:
        for line in(csvtick):
            tickers.append(line.strip('\n').strip('\r'))


        #print(tickers)
        #print('ABT' in tickers)

    df_ticks = pd.read_csv('input/tickers.csv')
    df_tdata = pd.read_csv('input/ticker_data.csv')
    pos_titles = []
    neg_titles = []
    words_in_HL_pos = []
    words_in_HL_neg = []
    stock_period_data = {}
    stock_period_date = []

    with open('input/title_energy.csv') as csvtitles:
        for line in(csvtitles):

            stock_attr = line.split(',')

            #check energy scalar
            #if stock_attr[-4] == '1':
            pos_titles.append(stock_attr)
            words_in_HL_pos.append(stock_attr[0].split(' '))
            for word in stock_attr[0].split(' '):
                
                #find stock tick mentions
                if word in tickers:
                    stock = word

                    #convert utc time of created reddit post to match stock value date
                    utc_date = float(stock_attr[-1].strip('\n'))
                    local_date = datetime.datetime.fromtimestamp(utc_date)
                    r_post_date = str(local_date.date())

                    #in case reddit post published on weekend, read stock value 2 days later (86400 secs per day)
                    if df_tdata[df_tdata['Date'].str.contains(str(r_post_date))].empty:
                        local_date = datetime.datetime.fromtimestamp(utc_date+86400*2)
                        r_post_date = local_date.date()

                    ind = df_tdata.index[df_tdata['Date'].str.contains(str(r_post_date))].to_list()[0]
                    
                    stock_period_date = [datetime.datetime.fromtimestamp(utc_date+86400*day).date() for day in range(control_period)]
                    
                    stock_period_data[stock + f' Energy of the post is {stock_attr[1]}'] = [df_tdata.iloc[ind:ind+control_period][stock].to_list(), stock_attr, stock_period_date]
                    
                    
    return stock_period_data
                    

data = XY_mapping()

def data_processing(data):
    ''' df_x = pd.read_csv(X_data)
    df_y = pd.read_csv(Y_data)
    del df_y["Date"]
    #print(df_y.values)
    df_y = df_y.to_numpy().transpose() '''
    
    Y_data = []
    X_data = []
    for title, xy_data in data.items():
        Y_data.append(xy_data[0][1:])
        X_data.append(xy_data[1][1:3] + [xy_data[0][0]]) # Add first value (day 1 - same day as reddit post) of stock as input
    
    print(X_data)
    scaler = preprocessing.StandardScaler().fit(Y_data)
    Y_scaled = scaler.transform(Y_data)
    print(Y_scaled)
    
    return X_data, Y_data


X,y = data_processing(data)

def ML_process(X, y):
     regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
     regr.predict(X[[0]])


#ML_process(X,y)
import csv
from pyexpat import model
from re import A
from turtle import shape
from numpy import append
from yfinance import ticker
import pandas as pd
import datetime
from sklearn import preprocessing
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def XY_mapping(control_period = 30):

    #list of stock tickers
    tickers = []
    with open('input/tickers.csv', newline = '') as csvtick:
        for line in(csvtick):
            tickers.append(line.strip('\n').strip('\r'))

    df_ticks = pd.read_csv('input/tickers.csv')
    df_tdata = pd.read_csv('input/ticker_data.csv')
    stock_period_data = {}
    stock_period_date = []


    df_inputdata = pd.read_csv('input/title_energy.csv')
    for index, title in enumerate(df_inputdata["title"]):
        
        title_words = title.split()
        for word in title_words:
            #find stock tick mentions
            if word in tickers:
                stock = word
                
                #convert utc time of created reddit post to match stock value date
                utc_date = df_inputdata["creation_date"][index]
                local_date = datetime.datetime.fromtimestamp(utc_date)
                r_post_date = str(local_date.date())

                #in case reddit post published on weekend, read stock value 2 days later (86400 secs per day)
                if df_tdata[df_tdata['Date'].str.contains(str(r_post_date))].empty:
                    local_date = datetime.datetime.fromtimestamp(utc_date+86400*2)
                    r_post_date = local_date.date()

                ind = df_tdata.index[df_tdata['Date'].str.contains(str(r_post_date))].to_list()[0]        
                stock_period_date = [datetime.datetime.fromtimestamp(utc_date+86400*day).date() for day in range(control_period)]
                
                stock_period_data[stock + f' Energy of the post is {df_inputdata["compound"][index]}'] = [
                    df_tdata.iloc[ind:ind+control_period][stock].to_list(),
                 df_inputdata.iloc[index].to_list() ]
                 
    return stock_period_data
                    

data = XY_mapping()

def data_processing(data):
    
    y_data = []
    X_data = []
    for title, xy_data in data.items():
        print("----",title,xy_data)
        y_data.append([float(val) for val in xy_data[0][1:]])

        if xy_data[1][3] == "Comment":
            xy_data[1][3] = 0.5

        reddit_scrape_info = [float(val) for val in xy_data[1][2:4]]
        title_energy_info = [float(val) for val in xy_data[1][5:]]

        X_data.append(reddit_scrape_info + [xy_data[0][0]] + title_energy_info)

         # Add first value (day 1 - same day as reddit post) of stock as input

    X_data=np.array(X_data)
    y_data=np.array(y_data)

    print(X_data,y_data)
    X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, test_size=0.33, random_state=42)
    
    print(X_train, "X_train", X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = data_processing(data)

def ML_process(X_train, X_test, y_train, y_test):
     #model_regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X_train, y_train)
     #model_regr.predict(X_test[[0]])
     #print(shape(X_train))

     print(np.shape(X_train), np.shape(y_train),'shape')
     model_linear = LinearRegression().fit(X_train, y_train)

     pred = model_linear.predict(X_test)
     
     return pred


y_pred = ML_process(X_train, X_test, y_train, y_test)

def plot(y_pred, y_test):

    print(y_pred, y_test)
    plt.figure()
    plt.plot(y_pred[0])
    plt.plot(y_test[0])
    plt.show()


plot(y_pred, y_test)
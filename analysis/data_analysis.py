import csv
from pyexpat import model
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import svm
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error

def XY_mapping(control_period = 30):

    #list of stock tickers
    tickers = []
    with open('input/tickers.csv', newline = '') as csvtick:
        for line in(csvtick):
            tickers.append(line.strip('\n').strip('\r'))
    tickers_full = []    
    with open('input/tickers_fullname.csv', newline = '') as csvtick_full:
        for line in(csvtick_full):
            tickers_full.append(line.strip('\n').strip('\r'))

    df_tdata = pd.read_csv('input/ticker_data.csv')
    stock_period_data = {}
    stock_period_date = []
    df_inputdata = pd.read_csv('input/title_energy.csv')
    
    for index, title in enumerate(df_inputdata["title"]):
        
        title_words = title.split()
        
        for word in title_words:
            #find stock tick mentions
            if word in tickers or word in tickers_full:    
                if word in tickers:
                    stock = word
                elif word in tickers_full:
                    index_ticker = tickers_full.index(word)
                    stock = tickers[index_ticker]
                
                #convert utc time of created reddit post to match stock value date
                utc_date = df_inputdata["creation_date"][index]   
                local_date = datetime.datetime.fromtimestamp(utc_date)
                r_post_date = str(local_date.date())
                print(r_post_date)
            

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
        
       # y_data.append(([float(val) for val in xy_data[0][1:]]))
        y_data.append(xy_data[0][1:])
        if xy_data[1][3] == "Comment":
            xy_data[1][3] = 0.5

        reddit_scrape_info = [float(val) for val in xy_data[1][2:4]]
        title_energy_info = [float(val) for val in xy_data[1][5:]]

        X_data.append(reddit_scrape_info + [xy_data[0][0]] + title_energy_info) # Add first value (day 1 - same day as reddit post) of stock as input

    
    print(y_data, X_data)
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = data_processing(data)

def ML_process(X_train, X_test, y_train, y_test, number_of_epochs = 20, batch_size = 10):
    print(np.shape(X_train), np.shape(y_train),'shape')

 
    X_train_LSTM = X_train.reshape((X_train.shape[0], 1, 7))
    X_test_LSTM = X_test.reshape((X_test.shape[0], 1, 7))
 
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(50, input_shape=(1, 7)))
    model_LSTM.add(Dense(29))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam')
    model_LSTM.fit(X_train_LSTM, y_train, epochs=100, verbose=0)
    model_RFregr = RandomForestRegressor().fit(X_train, y_train)
    #model_ARIMA = ARIMA(X_train, order = (1,1,0)).fit(X_train, y_train)
    model_linear = LinearRegression().fit(X_train, y_train)


    models = [model_RFregr,  model_linear]
    model_names= [ "LSTM Sequential" , "Random Forest Regressor", "Linear Regression"]
    y_pred = [model_LSTM.predict(X_test_LSTM)]
    for predictor in models:
        y_pred.append(predictor.predict(X_test))
     
     
    return y_pred, model_names 


y_pred, model_names = ML_process(X_train, X_test, y_train, y_test)

def model_evaluation(y_pred, y_test, model_names):
    for j, prediction in enumerate(y_pred):
        test_expected = np.concatenate(y_test)
        prediction = np.concatenate(prediction)
        mse = mean_squared_error(test_expected, prediction)
        r2 = r2_score(test_expected, prediction)
        mae = mean_absolute_error(test_expected, prediction)
        medae = median_absolute_error(test_expected, prediction)
        rmse = np.sqrt(mse)
        print('%s Test MSE: %.3f' % (model_names[j], mse))
        print('%s Test R2: %.3f' % (model_names[j], r2))
        print('%s Test MAE: %.3f' % (model_names[j], mae))
        print('%s Test MedAE: %.3f' % (model_names[j], medae))
        print('%s Test RMSE: %.3f' % (model_names[j], rmse))

model_evaluation(y_pred, y_test, model_names)

def plot(y_pred, y_test, model_names):
   # Iterate over the samples
    for i in range(len(y_test)):
        plt.figure()
        plt.plot(y_test[i], label="Expected")
        # Iterate over the models
        for j, prediction in enumerate(y_pred):
            
            plt.plot(prediction[i],label="Predicted {}".format(model_names[j]))
        plt.legend()
        plt.title("Sample {}".format(i))
        plt.show()
    
''' for j, prediction in enumerate(y_pred):
        for i, test_expected in enumerate(y_test):
                
            plt.figure()
            plt.plot(test_expected, label="Expected")
            plt.plot(prediction[i],label="Predicted")
            plt.legend()
            plt.title(model_names[j])
            plt.show()'''


plot(y_pred, y_test, model_names)
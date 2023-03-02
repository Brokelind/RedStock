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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, GRU, Flatten
from keras.wrappers.scikit_learn import KerasRegressor

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


def create_LSTM(X_train= X_train.reshape((X_train.shape[0], 1, 7)) ):
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(29))
    model.compile(optimizer='adam', loss='mse')
    return model
    
def optimize_LSTM(X_train, y_train):
    
    # Wrap the Keras model inside a KerasRegressor object
    model_LSTM = KerasRegressor(build_fn=create_LSTM, verbose=0)
    
    # Define the hyperparameters to tune
    param_grid = {'batch_size': [10, 20, 30, 40, 50],
                  'epochs': [10, 50, 100]}
    
    # Use grid search to find the best hyperparameters
    grid = GridSearchCV(estimator=model_LSTM, param_grid=param_grid, scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train, y_train)
    
    # Extract the best model
    best_model = grid_result.best_estimator_
    print(f"Best Parameters: {grid_result.best_params_}")
    return best_model

def optimize_CNN(X_train, y_train):
    # Define the CNN model
    model_CNN = Sequential()
    model_CNN.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model_CNN.add(MaxPooling1D(pool_size=2))
    model_CNN.add(Dropout(0.2))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(1))
    model_CNN.compile(loss='mean_squared_error', optimizer='adam')
    
    # Define the hyperparameters to tune
    param_grid = {'batch_size': [10, 20, 30, 40, 50],
                  'epochs': [10, 50, 100]}
    
    # Use grid search to find the best hyperparameters
    grid = GridSearchCV(estimator=model_CNN, param_grid=param_grid, scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train)
    
    # Extract the best model
    best_model = grid_result.best_estimator_
    print(f"Best Parameters: {grid_result.best_params_}")
    return best_model

def optimize_GRU(X_train, y_train):
    # Define the GRU model
    model_GRU = Sequential()
    model_GRU.add(GRU(50, input_shape=(1, X_train.shape[2])))
    model_GRU.add(Dense(1))
    model_GRU.compile(loss='mean_squared_error', optimizer='adam')
    
    # Define the hyperparameters to tune
    param_grid = {'batch_size': [10, 20, 30, 40, 50],
                  'epochs': [10, 50, 100]}
    
    # Use grid search to find the best hyperparameters
    grid = GridSearchCV(estimator=model_GRU, param_grid=param_grid, scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train, y_train)
    
    # Extract the best model
    best_model = grid_result.best_estimator_
    print(f"Best Parameters: {grid_result.best_params_}")
    return best_model
          
          

def ML_process(X_train, X_test, y_train, y_test, number_of_epochs = 20, batch_size = 10):
  
 
    # Reshape the input data for DNN
    X_train_DNN = X_train.reshape((X_train.shape[0], 1, 7))
    X_test_DNN = X_test.reshape((X_test.shape[0], 1, 7))
    print(X_train_DNN.shape)
    
    model_LSTM = optimize_LSTM(X_train_DNN, y_train)
    
    '''model_LSTM = optimize_LSTM(X_train_DNN, y_train)
    model_CNN = optimize_CNN(X_train, y_train)
    model_GRU = optimize_GRU(X_train_DNN, y_train)
    
    
    # Train and fit the models
    models = [model_LSTM, model_CNN, model_GRU]
    model_names= ["LSTM", "CNN", "GRU", "Random Forest Regressor", "Linear Regression"]
    y_pred = [model_LSTM.predict(X_test_DNN), model_CNN.predict(X_test_DNN), model_GRU.predict(X_test_DNN)]
    '''
    model_names= ["LSTM", "Random Forest Regressor", "Linear Regression"]
    y_pred = [model_LSTM.predict(X_test_DNN)]
    
    model_RFregr = RandomForestRegressor().fit(X_train, y_train)
    #model_ARIMA = ARIMA(X_train, order = (1,1,0)).fit(X_train, y_train)
    model_linear = LinearRegression().fit(X_train, y_train)
    models = [model_RFregr,  model_linear]
    
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
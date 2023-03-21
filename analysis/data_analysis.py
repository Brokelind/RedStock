import csv
from pyexpat import model
from re import A
from numpy import append
from yfinance import ticker
import pandas as pd
import datetime
from sklearn import preprocessing
import matplotlib.dates as mdates
import datetime
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

def XY_mapping(control_period=30):
    import pandas as pd
    import datetime

    # list of stock tickers
    tickers = []
    with open('input/tickers.csv', newline='') as csvtick:
        for line in (csvtick):
            tickers.append(line.strip('\n').strip('\r'))
    tickers_full = []
    with open('input/tickers_fullname.csv', newline='') as csvtick_full:
        for line in (csvtick_full):
            tickers_full.append(line.strip('\n').strip('\r'))

    df_tdata = pd.read_csv('input/ticker_data.csv')
    df_inputdata = pd.read_csv('input/title_energy.csv')
    stock_period_data = {}
    stock_period_date = {}
   

    for index, title in enumerate(df_inputdata["title"]):

        title_words = title.split()

        for word in title_words:
            # find stock tick mentions
            if word in tickers or word in tickers_full:
                if word in tickers:
                    stock = word
                elif word in tickers_full:
                    index_ticker = tickers_full.index(word)
                    stock = tickers[index_ticker]

                # convert utc time of created reddit post to match stock value date
                utc_date = df_inputdata["creation_date"][index]
                local_date = datetime.datetime.fromtimestamp(utc_date)
                r_post_date = pd.to_datetime(local_date.date())  # convert to datetime object

                counter = 0  # initialize counter
                # in case reddit post published on stock closed day, read stock value subsequent open stock day (86400 secs per day)
                while df_tdata.loc[df_tdata['Date'].str.contains(str(r_post_date))].empty:
                    if counter > control_period:
                        break  # break loop if running for too long
                    utc_date += 86400
                    r_post_date = pd.to_datetime(datetime.datetime.fromtimestamp(utc_date).date())
                    counter += 1

                ind = df_tdata.index[df_tdata['Date'].str.contains(str(r_post_date))].to_list()[0]
                #stock_period_date = [datetime.datetime.fromtimestamp(utc_date + 86400 * day).date() for day in range(control_period)]
                #print(stock_period_date)
                
                # Some stocks do not conatin data for the whole observed period
                if len(df_tdata.iloc[ind:ind + control_period][stock].to_list()) == 30:
                    stock_period_data["Stock obeserved: "+stock + f'\n Reddit post: {title} \n  Energy: positive:{df_inputdata["pos"][index]} negative:  {df_inputdata["neg"][index]}'] = [
                        df_tdata.iloc[ind:ind + control_period][stock].to_list(),
                        df_inputdata.iloc[index].to_list()]
                    
                    stock_period_date["Stock obeserved: "+stock + f'\n Reddit post: {title} \n  Energy: positive:{df_inputdata["pos"][index]} negative:  {df_inputdata["neg"][index]}'] = [
                        df_tdata.iloc[ind+1:ind + control_period]["Date"].to_list()]
                    
                    
    return stock_period_data, stock_period_date
                    

data, date = XY_mapping()
#print(date)

def data_processing(data):
    
    y_data = []
    X_data = []
    titles = []
    for title, xy_data in data.items():
        
        titles.append(title)
        # y_data.append(([float(val) for val in xy_data[0][1:]]))
        y_data.append(xy_data[0][1:])
        if xy_data[1][3] == "Comment":
            xy_data[1][3] = 0.5

        reddit_scrape_info = [float(val) for val in xy_data[1][2:4]]
        title_energy_info = [float(val) for val in xy_data[1][5:]]

        X_data.append(reddit_scrape_info + [xy_data[0][0]] + title_energy_info) # Add first value (day 1 - same day as reddit post) of stock as input

    
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    X_train, X_test, y_train, y_test, titles_train, titles_test = train_test_split( X_data, y_data, titles, test_size=0.2, random_state=42)
    
  
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test, titles_train, titles_test


X_train, X_test, y_train, y_test, titles_train, titles_test = data_processing(data)


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

def create_CNN_model():
    model_CNN = Sequential()
    model_CNN.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model_CNN.add(MaxPooling1D(pool_size=2))
    model_CNN.add(Dropout(0.2))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(y_train.shape[1]))
    model_CNN.compile(loss='mean_squared_error', optimizer='adam')
    return model_CNN

def optimize_CNN(X_train, y_train):
    # Define the KerasRegressor wrapper
    model = KerasRegressor(build_fn=create_CNN_model, verbose=0)
    
    # Define the hyperparameters to tune
    param_grid = {'batch_size': [10, 20, 30, 40, 50],
                  'epochs': [10, 50, 100]}
    
    # Use grid search to find the best hyperparameters
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train)
    
    # Extract the best model
    best_model = grid_result.best_estimator_.model
    print(f"Best Parameters: {grid_result.best_params_}")
    return best_model



'''def optimize_GRU(X_train, y_train):
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
           '''
          

def ML_process(X_train, X_test, y_train, y_test, number_of_epochs = 20, batch_size = 10):
    
    
    model_CNN = optimize_CNN(X_train,y_train)
   
    model_names= [ "CNN", "Random Forest Regressor", "Linear Regression"]
    y_pred = [model_CNN.predict(X_test)]
    
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



def plot(y_pred, y_test, model_names, titles, date):
    # Iterate over the samples
    for i in range(len(y_test)):
        # Extract dates from the date dictionary and convert to datetime objects
        date_list = date[titles[i]][0]
        print(date_list)
        date_list = [datetime.datetime.strptime(date_str[:10], '%Y-%m-%d') for date_str in date_list]
        
        plt.figure()
        
        plt.plot(date_list, y_test[i], label="Expected")
        # Iterate over the models
        for j, prediction in enumerate(y_pred):
            plt.plot(date_list, prediction[i], label="Predicted {}".format(model_names[j]))
            
        plt.legend()
        plt.title("Sample {}: {}".format(i, titles[i]))
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 5))
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.2)
        plt.show()


plot(y_pred, y_test, model_names, titles_test, date)
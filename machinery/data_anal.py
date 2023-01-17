import csv
from re import A
from numpy import append
from yfinance import ticker
import pandas as pd
import datetime

control_period = 30 #week days


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
stock_period_val = {}
stock_period_date = []

with open('input/Headline_energy.csv') as csvtitles:
    for line in(csvtitles):

        stock_attr = line.split(',')

        #check energy scalar
        if stock_attr[-4] == '1':
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

                    #stock_val_reaction = df_tdata.loc[df_tdata['Date'] == str(r_post_date), stock].to_list()
                    
                    print(df_tdata[df_tdata['Date'].str.contains(str(r_post_date))], r_post_date)
                    ind = df_tdata.index[df_tdata['Date'].str.contains(str(r_post_date))].to_list()[0]
                    
                    stock_period_date = [datetime.datetime.fromtimestamp(utc_date+86400*day).date() for day in range(control_period)]
                    print(df_tdata.iloc[ind:ind+control_period][stock],"asdasdas")
                    stock_period_val[stock + ' Positive'] = [df_tdata.iloc[ind:ind+control_period][stock].to_list(), stock_attr, stock_period_date]
                    

        elif stock_attr[-4] == '-1':
            neg_titles.append(stock_attr)
            words_in_HL_neg.append(stock_attr[0].split(' '))
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
                        
                    print(df_tdata[df_tdata['Date'].str.contains(str(r_post_date))])
                    
                    ind = df_tdata.index[df_tdata['Date'].str.contains(str(r_post_date))].to_list()[0]
                    
                    stock_period_date = [datetime.datetime.fromtimestamp(utc_date+86400*day).date() for day in range(control_period)]         
                    stock_period_val[stock +' Negative'] = [df_tdata.iloc[ind:ind+control_period][stock].to_list(), stock_attr, stock_period_date]
                

    #print(stock_period_val)


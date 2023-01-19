from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from data_analysis import stock_period_val

i = 1

for key, val in stock_period_val.items():
    print("----",key,"+++++", val)
    
    plt.figure()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 5))
    
    plt.plot(val[2], val[0])
    
    plt.text(val[2][0], val[0][0],'Posted to Reddit')
    
    plt.title(key)
    plt.suptitle(val[1][0])
    #i += 1
    plt.show()
    


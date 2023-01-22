import pandas as pd 
from yfinance import ticker
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
polarity_results = []

df_postinfo = pd.read_csv("input/post_info.csv")
headlines= df_postinfo["title"]
neu = []
pos = []
neg = []
compound = []

for post_title in headlines:
    pol_score = sia.polarity_scores(post_title) #dict
    neu.append(pol_score["neu"])
    pos.append(pol_score["pos"])
    neg.append(pol_score["neg"])
    compound.append(pol_score["compound"])

    pol_score['headline'] = post_title
    polarity_results.append(pol_score)
    

df_postinfo["neu"]=neu
df_postinfo["pos"]=pos
df_postinfo["neg"]=neg
df_postinfo["compound"]=compound
df_postinfo.to_csv("input/title_energy.csv")
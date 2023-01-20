from yfinance import ticker
from data_scrape import*
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
polarity_results = []

for line in headlines:
    pol_score = sia.polarity_scores(line) #dict
    pol_score['headline'] = line
    polarity_results.append(pol_score)


df = pd.DataFrame.from_records(polarity_results)
df.head()

upvote_ratio.pop(0)
upvote_amount.pop(0)
creation_date.pop(0)

'''
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
'''

df['upvotes'] = upvote_amount
df['upvote_ratio'] = upvote_ratio
df['utc_time'] = creation_date
print(df)


df2 = df[['headline', 'label', 'upvotes', 'upvote_ratio', 'utc_time']]
df2.to_csv('input/title_energy.csv',header=False, encoding='utf-8', index=False)

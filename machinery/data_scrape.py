
from re import S
import praw
import numpy as np
import pandas as pd
from id_info import *


reddit = praw.Reddit(client_id = client_id,
client_secret = client_secret,
user_agent = user_agent)

headlines = set()

upvote_amount = []
upvote_ratio = []
creation_date = []


#https://praw.readthedocs.io/en/latest/code_overview/models/submission.html
#hot new rising top
for submission in reddit.subreddit('stocks').hot(limit=None):
    #print(submission.title)
    headlines.add(submission.title)
    upvote_amount.append(submission.score)
    upvote_ratio.append(submission.upvote_ratio)
    creation_date.append(submission.created_utc)
#print(len(headlines))
#print(upvote_amount)


df = pd.DataFrame(headlines)
#df.head()

df.to_csv('input/headlines.csv', header = False, encoding="utf-8",
index=False)


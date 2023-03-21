import praw
import pandas as pd
from id_info import *
import csv

reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

submissions = []
for submission in reddit.subreddit('stocks').search(query='timestamp:2022',
                                                    sort='top',
                                                    limit=200):
    if not submission.stickied:
        submissions.append(submission)

df_full = pd.DataFrame({
    'title': [s.title for s in submissions],
    'upvote_amount': [s.score for s in submissions],
    'upvote_ratio': [s.upvote_ratio for s in submissions],
    'creation_date': [s.created_utc for s in submissions]
})


df_full.to_csv('input/post_info.csv', encoding='utf-8', index=False)

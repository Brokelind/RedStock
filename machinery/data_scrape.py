import praw
import pandas as pd
from id_info import *
import csv
from datetime import datetime, timedelta

with open('input/tickers.csv', newline="") as f:
    reader = csv.reader(f)
    stock_names = list(reader)
with open('input/tickers_fullname.csv', newline="") as f:
    reader = csv.reader(f)
    stock_fullnames = list(reader)

stock_names = [item for sublist in stock_names for item in sublist]
stock_fullnames = [item for sublist in stock_fullnames for item in sublist]
stock_name_search = stock_fullnames + stock_names


number_of_comments_to_include = 2

reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

headlines = set()
titles = []
upvote_amount = []
upvote_ratio = []
creation_date = []
post_comments = []

for submission in reddit.subreddit('stocks').top(time_filter='year',
                                    
                                                    limit = 2000):
    print(submission)
    
    # Skip if post was created within last 30 days
    if datetime.utcfromtimestamp(submission.created_utc) > datetime.utcnow() - timedelta(days=30) and datetime.utcfromtimestamp(submission.created_utc) < datetime.utcnow() - timedelta(days=365):
        continue
    
    if not submission.stickied:  # ignore stickied posts
        if any(title_word in submission.title for title_word in stock_name_search):
            print(submission.title, "stock mentioned!")

            headlines.add(submission.title)
            titles.append(submission.title)
            upvote_amount.append(submission.score)
            upvote_ratio.append(submission.upvote_ratio)
            creation_date.append(submission.created_utc)

        '''else:
            submission.comment_sort = "top"
            i = 1
            for comment in submission.comments.list():
                submission.comments.replace_more()
                if any(title_word in comment.body for title_word in stock_name_search):
                    headlines.add(comment.body)
                    titles.append(comment.body)
                    upvote_amount.append(comment.score)
                    upvote_ratio.append(0)
                    creation_date.append(comment.created_utc)
                    i += 1
                if i > number_of_comments_to_include:
                    break '''

df_titles = pd.DataFrame(headlines)

print(df_titles)
full_dict = {'title': titles, 'upvote_amount': upvote_amount,
             'upvote_ratio': upvote_ratio, 'creation_date': creation_date}

df_full = pd.DataFrame(full_dict)
df_full.to_csv('input/post_info.csv', encoding="utf-8", index=False)

df_titles.to_csv('input/titles.csv', header=False, encoding="utf-8", index=False)


from re import S
import praw
import numpy as np
import pandas as pd
from id_info import*

import csv

number_of_commments_to_include = 2

reddit = praw.Reddit(client_id = client_id,
client_secret = client_secret,
user_agent = user_agent)


with open('input/tickers.csv', newline="") as f:
    reader = csv.reader(f)
    stock_names = list(reader)
with open('input/tickers_fullname.csv', newline="") as f:
    reader = csv.reader(f)
    stock_fullnames = list(reader)
    
stock_names = [item for sublist in stock_names for item in sublist]
stock_fullnames =  [item for sublist in stock_fullnames for item in sublist]
stock_name_search = stock_fullnames + stock_names


headlines = set()
upvote_amount = []
upvote_ratio = []
creation_date = []
post_comments = []


#https://praw.readthedocs.io/en/latest/code_overview/models/submission.html
#hot new rising top
for submission in reddit.subreddit('stocks').hot(limit=2):
    
    if any(title_word in stock_name_search for title_word in submission.title.split()):
        print(submission.title, "stock mentioned!")

        headlines.add(submission.title)
        upvote_amount.append(submission.score)
        upvote_ratio.append(submission.upvote_ratio)
        creation_date.append(submission.created_utc)
        
    else:
        
        submission.comment_sort = "top"
        i = 1
        for comment in submission.comments:
            submission.comments.replace_more()
            if any(title_word in stock_name_search for title_word in comment.body.split()):
                
                headlines.add(comment.body)
                upvote_amount.append(comment.score)
                upvote_ratio.append("Comment")
                creation_date.append(comment.created_utc)
                i+=1
            if i > number_of_commments_to_include:
                break
               

df_titles = pd.DataFrame(headlines)

full_dict = {'title': headlines, 'upvote_amount': upvote_amount,
 'upvote_ratio': upvote_ratio, 'creation_date':creation_date}

df_full = pd.DataFrame(full_dict)
df_full.to_csv('input/post_info.csv', encoding="utf-8",
index=False)

df_titles.to_csv('input/titles.csv', header = False, encoding="utf-8",
index=False)
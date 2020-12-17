import numpy as np
import pandas as pd

def cleandata(df):
  
  # reviewText_iTem contains all reviews pertaining to that item
  # reviewText_user contains all reviews pertaining to that user
  # overall_item contains the average rating per item
  # overall_user contains the average rating per user
  df['reviewText_item'] = df.groupby(['asin'])['reviewText'].transform(lambda x : ' '.join(x)) 
  df['overall_item']=df.groupby('asin').overall.transform('mean').round()

  df['reviewText_user'] = df.groupby(['reviewerID'])['reviewText'].transform(lambda x : ' '.join(x)) 
  df['overall_user']=df.groupby('reviewerID').overall.transform('mean').round()

  df['reviewText_item'].dropna(inplace=True)
  df['reviewText_user'].dropna(inplace=True)
  
  
  #drops dpulicate data
  df=df.drop_duplicates(subset=['reviewText_item'])
  df=df.drop_duplicates(subset=['reviewText_user'])

  # overall_avg contains average rating per item per user
  cols = ['overall_item','overall_user']
  df['overall_avg'] = df[cols].astype(float).mean(axis=1)
  df['overall_avg'] = df['overall_avg'].round()
  df['overall_avg'] = df['overall_avg'].astype(int)
  
  # slices the data in half to work with a manageable amount of data
  df1=df.sample(frac=0.5)
  
  # normalizes overall rating
  zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
  df1['overall'] = df1['overall'].apply(lambda x: zero_numbering[x])
  df1['overall_avg'] = df1['overall_avg'].apply(lambda x: zero_numbering[x])
  
  return df1

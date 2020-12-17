import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

def cleandata(df):
  
  global asin_columns
  global reviewer_columns
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
  min_max_scaler = preprocessing.MinMaxScaler()
  x_item = df.overall_item.values 
  x_item_reshape = x_item.reshape(-1,1)
  x_item_scaled = min_max_scaler.fit_transform(x_item_reshape)
  normalized_overallitem = pd.DataFrame(x_item_scaled,columns=['normalized_item'])
  
  x_user = df.overall_user.values 
  x_user_reshape = x_user.reshape(-1,1)
  x_user_scaled = min_max_scaler.fit_transform(x_user_reshape)
  normalized_overalluser = pd.DataFrame(x_user_scaled,columns=['normalized_user'])

  
  df1 = pd.concat([df1, normalized_overallitem, normalized_overalluser],axis=1)

  return df1

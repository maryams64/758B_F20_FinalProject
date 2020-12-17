import pandas as pd
import numpy as np
from preprocessing.preprocessing import cleandata
from preprocessing.vectorization import get_counts, del_words, create_vocab, encode_sentence
from preprocessing.ReviewDataset import ReviewsDataset
from model.MultiInputModel import MultiInputModel
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

print("Reading data....")
asin_columns = []
reviewer_columns = []

df = pd.read_json('data/Digital_Music_5.json', lines = True)

print("Cleaning data....")
df1 = cleandata(df)

df1['reviewText_item'].astype('str')
df1['reviewText_user'].astype('str')

print("Vectorizing data....")
item_counts = get_counts(df1, 'reviewText_item')
user_counts = get_counts(df1, 'reviewText_user')

item_counts = del_words(item_counts)
user_counts = del_words(user_counts)

item_words = create_vocab(item_counts)
user_words = create_vocab(user_counts)

df1['encoded_item'] = df1['reviewText_item'].apply(lambda x: np.array(encode_sentence(x, item_words[1])))
df1['encoded_user']=df1['reviewText_user'].apply(lambda x: np.array(encode_sentence(x, user_words[1])))

print("Creating training and validation datasets....")
item_list = list(df1['encoded_item'])
itemrating_list = list(df1['normalized_item'])
asin_list = list(df1[df1.columns[asin_columns]])

item_list.expand(asin_list).expand(itemrating_list)

user_list = list(df1['encoded_user'])
userrating_list = list(df1['normalized_user'])
reviewerID_list = list(df1df1.columns[[reviewer_columns]])

user_list.expand(reviewerID_list).expand(userrating_list)

X= item_list
X2= user_list
y = list(df1['overall_avg'])

print(len(x))

'''X_item_train, X_item_valid, y_item_train, y_item_valid = train_test_split(X, y, test_size=0.2)
X_user_train, X_user_valid, y_user_train, y_user_valid = train_test_split(X2, y, test_size=0.2)

train_item_ds = ReviewsDataset(X_item_train, y_item_train)
valid_item_ds = ReviewsDataset(X_item_valid, y_item_valid)

train_user_ds = ReviewsDataset(X_user_train, y_user_train)
valid_user_ds = ReviewsDataset(X_user_valid, y_user_valid)

batch_size = 5000
item_vocab = len(item_words[0])
user_vocab = len(user_words[0])
item_train_dl = DataLoader(train_item_ds, batch_size=batch_size)
item_valid_dl = DataLoader(valid_item_ds, batch_size=batch_size)

user_train_dl = DataLoader(train_user_ds, batch_size=batch_size)
user_valid_dl = DataLoader(valid_user_ds, batch_size=batch_size)

print("Running MultiInputModel....")
MultiInputModel(item_train_dl,item_valid_dl,item_vocab,user_train_dl,user_valid_dl,user_vocab,100,70,y)'''

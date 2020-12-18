from model.LSTM import LSTM
from training.LSTMtrain import train_LSTMmodel
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

##Functions that will seek the best model + load it and get final output
def getLSTMModel(item_train_dl,item_valid_dl,item_vocab,user_train_dl,user_valid_dl,user_vocab,embedding_dim,hidden_dim):
  print('Finding the best LSTM Models....')
  lstm_item_model = LSTM(item_vocab, embedding_dim, hidden_dim)
  lstm_user_model = LSTM(user_vocab, embedding_dim, hidden_dim)

  lstm_item_output = train_LSTMmodel(lstm_item_model, item_train_dl, item_valid_dl, 1, epochs=30, lr=0.01)
  lstm_user_output = train_LSTMmodel(lstm_user_model, user_train_dl, user_valid_dl, 2, epochs=30, lr=0.01)

  item_model_PATH = 'model/model1.pt'
  user_model_PATH = 'model/model2.pt'

  item_LSTMmodel = LSTM(item_vocab, embedding_dim, hidden_dim)
  user_LSTMmodel = LSTM(user_vocab, embedding_dim, hidden_dim)

  item_LSTMmodel.load_state_dict(torch.load(item_model_PATH))
  user_LSTMmodel.load_state_dict(torch.load(user_model_PATH))

  return item_LSTMmodel, user_LSTMmodel

def LSTMoutput(item_LSTMmodel, user_LSTMmodel, item_train_dl, user_train_dl):
  print("Getting LSTM output....")
  item_LSTMmodel.eval()
  user_LSTMmodel.eval()

  models = [item_LSTMmodel, user_LSTMmodel]
  dataloaders = [item_train_dl, user_train_dl]
  tensors = []

  for i in range(len(models)):
    for x, y, l in dataloaders[i]:
      x = x.long()
      y = y.long()
      y_hat = models[i](x, l)
    tensors.append(y_hat)
  
  FCFN_input = torch.cat([tensors[0],tensors[1]],dim=1)
  return FCFN_input

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

valid_scores = []
#defining the functions which will find the best LSTM Model
def train_LSTMmodel(model, dataloader, val, num, epochs=10, lr=0.001):
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    max = []
    metrics = []
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for X, Y, h in dataloader:
            X = X.long()
            Y = Y.long()
            Y_pred = model(X, h)
            optimizer.zero_grad()
            loss = F.cross_entropy(Y_pred, Y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*Y.shape[0]
            total += Y.shape[0]
        val_loss, val_acc, val_rmse = validation_LSTMmetrics(model, val)
        valid_scores.append([i,val_loss, val_acc, val_rmse])
        if i % 5 == 1:
            #print("epoch: %.1f: train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (i, sum_loss/total, val_loss, val_acc, val_rmse))
            modelSave = savebestmodel(val_acc, max, metrics)
            if modelSave:
              torch.save(model.state_dict(), 'model/model'+str(num)+'.pt')
    valid_data = pd.Datagrame(valid_scores,columns=['Epoch','Valid_Loss','Valid_Accuracy','Valid_RMSE'])
    valid_data.set_index('Epoch',inplace=True)
    plt.figure(figsize=(10, 10))
    plt.plot(valid_data[0], valid_data[1:])
    plt.savefig("model_scores.png")
    
    return Y_pred
    
def validation_LSTMmetrics (model, val):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for X, Y, h in val:
        X = X.long()
        Y = Y.long()
        Y_hat = model(X, h)
        loss = F.cross_entropy(Y_hat, Y)
        pred = torch.max(Y_hat, 1)[1]
        correct += (pred == Y).float().sum()
        total += Y.shape[0]
        sum_loss += loss.item()*Y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, Y.unsqueeze(-1)))*Y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total

def savebestmodel(val_acc, max, metrics):
  #compares validation accuracy --> can change so that it compares training accuracy, or compares any combination of loss, accuracy and RMSE
  modelSave = False
  metrics.append(val_acc)
  if len(metrics) > 1:
    i = 0
    while i+1 < len(metrics):
      if metrics[i+1] > max[0]:
        modelSave = True
        max[0] = metrics[i+1]
      else:
        modelSave = False
      i += 1
  else:
    modelSave = True
    max.append(metrics[0])
  return modelSave

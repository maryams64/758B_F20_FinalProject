import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

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
        for x, y, l in dataloader:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]    
        val_loss, val_acc, val_rmse = validation_LSTMmetrics(model, val)
        if i % 5 == 1:
            #print("epoch: %.1f: train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (i, sum_loss/total, val_loss, val_acc, val_rmse))
            modelSave = savebestmodel(val_acc, max, metrics)
            if modelSave:
              torch.save(model.state_dict(), '/content/drive/MyDrive/758B/Big Data Project/model'+str(num)+'.pt')
    return y_pred
    
def validation_LSTMmetrics (model, val):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in val:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
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


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

#FCFN functions
def trainFCFN(model, train_loader, optimizer, criterion):

  epoch_loss, epoch_acc = 0.0, 0.0
  model.train()

  for batch_idx,(data, target) in enumerate(train_loader):
    #print(data)
    data, target = Variable(data,volatile=True), Variable(target)

    model = model.float()
    data = data.float()
    target = target.long()

    #zero gradient
    optimizer.zero_grad()
    
    #get the predicted outputs for the batch_x
    predictions = model(data)
    #predictions = predictions.detach().numpy()

    #loss = calculate the loss using the predicted output and the truth
    loss = criterion(predictions, target)
  
    #acc = calculate the accuracy for this batch_x
    pred = predictions.data.max(1)[1]
    acc = pred.eq(target.data).sum()
 
    #backpropogate loss.backward()
    loss.backward()
    #update parameters optimizer.step()
    optimizer.step()

    epoch_loss += loss.item()
    epoch_acc += acc.item()

    #calculate average loss and average accuracy for this epoch
    avg_loss = epoch_loss/len(train_loader.dataset)
    avg_acc = epoch_acc/len(train_loader.dataset)
  
  return avg_loss, avg_acc
def evaluateFCFN(model, val_loader, criterion):
  epoch_data = []
  epoch_loss, epoch_acc = 0.0, 0.0
  model.eval()

  with torch.no_grad():
    for data, target in val_loader:
      data, target = Variable(data,volatile=True), Variable(target)

      model = model.float()
      data = data.float()

      predictions = model(data)

      #print(predictions.dtype)  

      epoch_loss += criterion(predictions, target).item() 
      pred = predictions.data.max(1)[1]
      epoch_acc += pred.eq(target.data).sum().item()

      avg_loss = epoch_loss/len(val_loader.dataset)
      avg_acc = epoch_acc/len(val_loader.dataset)
      epoch_data.append([avg_loss,avg_acc])
      epoch_df = pd.DataFrame(epoch_data,columns=['Avg. Loss','Avg. Accuracy'])
      epoch_df.to_csv("758B_F20_FinalProject/epoch_data.csv")
  

  return avg_loss, avg_acc

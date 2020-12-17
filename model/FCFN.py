import torch
import torch.nn as nn 
import torch.nn.functional as F 

class FCFN(nn.Module):

  def __init__(self,input_dim=14000,output_dim=5, dropout_rate=0.5):
    super(FCFN,self).__init__()
    self.fc1 = nn.Linear(input_dim,7000)
    self.fc2 = nn.Linear(7000, 1000)
    self.fc3 = nn.Linear(1000, 500)
    self.fc4 = nn.Linear(500, output_dim)
    self.dropout = nn.Dropout(dropout_rate)
  def forward(self, x):
    print(f"FCFN input: {x.shape}")
    x = F.relu(self.fc1(x))    
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = F.relu(self.fc3(x))
    x = self.dropout(x)
    x = F.log_softmax(self.fc4(x), dim =1)

    return x

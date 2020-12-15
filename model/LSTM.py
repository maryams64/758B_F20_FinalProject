import torch
import torch.nn as nn 
import torch.nn.functional as F 

class LSTM_fixed_len2(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        x, hidden = self.lstm(x)
        x = x.contiguous().view(-1,self.num_flat_features(x))
        return x
        
    def num_flat_features(self,x):
       size = x.size()[1:]
       num_features = 1
       
       for s in size:
         num_features*= s
         
       return num_features

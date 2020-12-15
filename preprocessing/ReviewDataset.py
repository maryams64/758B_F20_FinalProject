from torch.utils.data import Datase
import torch
import numpy as np

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.Y[idx], self.X[idx][1]

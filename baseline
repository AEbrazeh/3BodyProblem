import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

class baselineNN(nn.Module):
    def __init__(self, hiddenLayers, hiddenDim):
        super(baselineNN, self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(3, hiddenDim),
                                     nn.ReLU())
        for _ in range(hiddenLayers - 1):
            self.layers.append(nn.Linear(hiddenDim, hiddenDim))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hiddenDim, 4))
            
    def forward(self, x):
        x0 = torch.zeros_like(x[:, :2], device=x.device)
        x0[:, 0] = 1
        
        r = torch.cat((x0, x[:, :2]), dim=-1)
        return self.layers(x) + r
    
    def load(self, file):
        self.load_state_dict(torch.load(file))
        
    def save(self, file):
        torch.save(self, file)
    
class SimpleDataset(Dataset):
    def __init__(self, dataLoc):
        Data = pd.read_parquet(dataLoc).to_numpy().reshape((-1, 2561, 4))
        self.xData = np.concatenate((Data[:, None, 0, 2:].repeat(2561, 1), np.zeros((len(Data), 2561, 1))), -1)
        self.xData[:, :, -1] = np.linspace(0, 10, 2561)
        self.xData = self.xData.reshape(-1, 3)
        self.yData = Data.reshape(-1, 4)
        del(Data)
        
    def __len__(self):
        return self.xData.shape[0]
    
    def __getitem__(self, index):
        return self.xData[index], self.yData[index]

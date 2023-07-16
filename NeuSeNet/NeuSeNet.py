import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class multiLinear(nn.Module):
    def __init__(self, inputDim, outputDim, numLayers, applyActivation=True):
        super(multiLinear, self).__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.numLayers = numLayers
        w = torch.randn(numLayers, outputDim, inputDim) / np.sqrt(inputDim)
        b = torch.zeros((numLayers, 1, outputDim))
        
        self.weights = nn.Parameter(w, requires_grad=True)
        self.bias = nn.Parameter(b, requires_grad=True)
        
        if applyActivation == True:
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.Identity()
            
    def forward(self, x):
        if len(x.shape) == 2:
            return self.activation(torch.einsum('Noi, bi -> Nbo', self.weights, x) + self.bias)
        
        if len(x.shape) == 3:
            return self.activation(torch.einsum('Noi, Nbi -> Nbo', self.weights, x) + self.bias)
        
class multiFeedForward(nn.Module):
    def __init__(self, inputDim, outputDim, hiddenDim, numHiddenLayers, numNets):
        super(multiFeedForward, self).__init__()
        self.Model = nn.Sequential(multiLinear(inputDim, hiddenDim, numNets))
        for _ in range(numHiddenLayers-1):
            self.Model.append(multiLinear(hiddenDim, hiddenDim, numNets))
        self.Model.append(multiLinear(hiddenDim, outputDim, numNets, False))

    def forward(self, x):
        y = self.Model(x)
        return y
    
class SeriesNet(nn.Module):
    def __init__(self, numHiddenLayers, hiddenDim, order, basis="polynomial"):
        super(SeriesNet, self).__init__()
        self.order = order
        self.basis = basis
        if self.basis == 'polynomial':
            self.Layers = multiFeedForward(2, order, hiddenDim, numHiddenLayers, 4)
            
        if self.basis == 'harmonic':
            self.Layers = multiFeedForward(2, 2 * (order-1), hiddenDim, numHiddenLayers, 4)
        
    def forward(self, x, t, deriv=False, scale = 1):
        x0 = torch.zeros_like(x, device=x.device)
        x0[:, 0] = 1
        

        if self.basis == 'polynomial':
            r = torch.cat((x0, x), dim=1).unsqueeze(0)
            phi = torch.cat((r, self.Layers(x).permute(2, 1, 0)), dim = 0)
            n = torch.arange(self.order + 1, device=x.device)
            t_ = (t/scale).unsqueeze(-1)**n / torch.lgamma(n+1).exp()        
            y = torch.einsum('Nbo, bN -> bo', phi, t_)
            if deriv:
                dy = torch.einsum('Nbo, bN -> bo', phi[1:], t_[:, :-1] / scale)
                return y, dy
            return y
            
        if self.basis == 'harmonic':
            r = torch.cat((x0, x), dim=1)
            n = torch.arange(self.order, device=x.device) + 1
            S = torch.sin(np.pi * n * t.unsqueeze(-1) / scale)
            C = torch.cos(np.pi * n * t.unsqueeze(-1) / scale)
            
            phi_ = self.Layers(x).permute(2, 1, 0)
            phi_ = phi_.reshape(2, self.order-1, *phi_.shape[1:])
            phi = torch.cat((-(phi_[0] * n[1:, None, None]).sum(0).unsqueeze(0), phi_[0], -phi_[1].sum(0).unsqueeze(0), phi_[1]), dim = 0)       
            
            y = torch.einsum('Nbo, bN -> bo', phi, torch.cat((S, C), dim=-1)) + r

            if deriv:
                dy = torch.einsum('Nbo, bN -> bo', phi, torch.cat((np.pi * n * C / scale, -np.pi * n * S / scale), dim=-1))
                return y, dy
            return y
    
    def load(self, file):
        self.load_state_dict(torch.load(file))
        
    def save(self, file):
        torch.save(self, file)
        
        
class SimpleDataset(Dataset):
    def __init__(self, dataLoc):
        Data = pd.read_parquet(dataLoc).to_numpy().reshape((-1, 2561, 4))
        self.xData = Data[:, None, 0, 2:].repeat(2561, 1).reshape(-1, 2)
        self.yData = Data.reshape(-1, 4)
        self.tData = np.linspace(0, 10, 2561).reshape(1, -1).repeat(len(Data), 0).flatten()
        del(Data)
        
    def __len__(self):
        return self.yData.shape[0]
    
    def __getitem__(self, index):
        return self.xData[index], self.tData[index], self.yData[index]

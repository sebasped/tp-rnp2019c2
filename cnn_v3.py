#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:18:28 2019

@author: sebas
"""

import torch
import torchvision as tv
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


class CNN(torch.nn.Module):
    def __init__(self, in_chann=1, out_classes=2): 
        super().__init__()
        self.c1 = torch.nn.Conv1d(in_chann, out_channels=5, kernel_size=3, stride=1, padding=1)
#        self.c2 = torch.nn.Conv2d(5, 25, 3, 1, 1)
        self.c3 = torch.nn.MaxPool1d(10)
        self.H = 5*150
#        self.L1 = torch.nn.Linear(self.H, 512)
#        self.L2 = torch.nn.Linear(512,out_classes)
        self.L1 = torch.nn.Linear( self.H, out_classes)

    def forward(self, x):
#        y1 = self.c1(x).relu()
#        y2 = self.c2(y1)
#        y3 = self.c3(y2).relu()
#        y4 = self.L1(y3.view(-1,self.H)).tanh()
#        y5 = self.L2(y4)

        y1 = self.c1(x)
#        y2 = self.c2(y1)
        y3 = self.c3(y1).relu()
        y4 = self.L1(y3.view(-1,self.H))
#        y5 = y4.softmax(dim=1)
        y5=y4

        return y5



if __name__ == '__main__':

    
    sinPico=[]
    path = './'
    filename = 'sinPico.txt'
    with open(path+filename, 'r') as f:
       lines = (line.strip() for line in f if line)
    #       xtest = [int(float(line)) for line in lines]
       for line in lines:
           floats = [float(x) for x in line.split()]
           sinPico.append(floats)
    
    conPico=[]
    path = './'
    filename = 'conPico.txt'
    with open(path+filename, 'r') as f:
       lines = (line.strip() for line in f if line)
    #       xtest = [int(float(line)) for line in lines]
       for line in lines:
           floats = [float(x) for x in line.split()]
           conPico.append(floats)
    
    n=50
    sinPico_trn = sinPico[0:n]
    sinPico_tst = sinPico[n:2*n]

    conPico_trn = conPico[0:n]
    conPico_tst = conPico[n:2*n]

    series_trn = sinPico_trn + conPico_trn
    series_tst = sinPico_tst + conPico_tst
    
    data_trn = torch.tensor(series_trn).view(len(series_trn),1,1500)
    labels_trn = torch.tensor([0]*n+[1]*n)#.view(11,1)
    
    data_tst = torch.tensor(series_tst).view(len(series_tst),1,1500)
    labels_tst = torch.tensor([0]*n+[1]*n)#.view(11,1)
    
    trn_data = TensorDataset( data_trn, labels_trn)
    tst_data = TensorDataset( data_tst, labels_tst)
    
    B=2
    trn_load = DataLoader( trn_data, shuffle=True, batch_size=B)
    tst_load = DataLoader( tst_data, shuffle=True, batch_size=B)


    model = CNN()
    optim = torch.optim.Adam(model.parameters())
    costf = torch.nn.CrossEntropyLoss()
    
    T = 20
    model.train()
    for t in range(T):
      E = 0
      for data, label in trn_load:
        optim.zero_grad()
        y = model(data)
        error = costf(y, label)
        error.backward()
        optim.step()
        E += error.item()
      print(t, E) 
      
      
    model.eval()
    right, total = 0, 0
    with torch.no_grad():
        for data, labels in tst_load:
            y = model(data)
            right += (y.argmax(dim=1)==labels).sum().item()
            total += len(labels)

    accuracy = right / total
    print('Accuracy: ', accuracy)
    
    
    
    
    

##
##
##"""
##puedo usar la red convolucional para hacer un autoencoder
##
##_.enc = torch.nn.Sequential(
##  conv2d(...),
##  torch.nn.ReLU(True))
##
##_.dec = torch.nn.ConvTranspose2d()
##Tanh()
#"""
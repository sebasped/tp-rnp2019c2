#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:18:28 2019

@author: sebas
"""


""" Primera versión final con todo:
    levanta datos y los particiona en train test
    Hace batch
    Acc 83-5% lo mejor que se obtuvo
    Arquitectura poco profunda
"""


import torch
import torchvision as tv
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time


class CNN(torch.nn.Module):
    def __init__(self, in_chann=1, out_classes=2): 
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_chann, out_channels=5, kernel_size=60, stride=2, padding=20)
        self.conv2 = torch.nn.Conv1d(5, 10, 30, 1, 10)
        self.mpool1 = torch.nn.MaxPool1d(4)
        self.H = 10*183
        self.Linear1 = torch.nn.Linear(self.H, 300)
#        self.Linear2 = torch.nn.Linear(400,100)
        self.Linear3 = torch.nn.Linear(300,out_classes)

    def forward(self, x):
        y1 = self.conv1(x).relu()
        y2 = self.conv2(y1).relu() #probas de poner el relu antes del pooling
        y3 = self.mpool1(y2)
        y4 = self.Linear1(y3.view(-1,self.H)).tanh()
#        y4 = self.Linear1(y3.view(-1,self.H))
        y5 = self.Linear3(y4)
#        y6 = self.Linear3(y5)

#        y1 = self.conv1(x).relu()
#        y2 = self.c2(y1)
#        y3 = self.mpool1(y1).relu()
#        y4 = self.Linear1(y3.view(-1,self.H))
#        y5 = y4.softmax(dim=1)
#        y5=y4

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
    
    #Para que queden las clases balanceadas
    N = min( len(sinPico), len(conPico))
    if len(sinPico) > N:
        sinPico = sinPico[0:N]
    if len(conPico) > N:
        conPico = conPico[0:N]    
    
    
    # 80/20 train/test
    sinPico_trn = sinPico[:-N//5]
    sinPico_tst = sinPico[-N//5:]

    conPico_trn = conPico[:-N//5]
    conPico_tst = conPico[-N//5:]

    series_trn = sinPico_trn + conPico_trn
    series_tst = sinPico_tst + conPico_tst
    
    cant_mediciones_por_dato = len( sinPico[0])
    data_trn = torch.tensor( series_trn).view(len(series_trn), 1, cant_mediciones_por_dato)
    labels_trn = torch.tensor( [0]*len(sinPico_trn)+[1]*len(conPico_trn))#.view(11,1)
    
    data_tst = torch.tensor( series_tst).view(len(series_tst), 1, cant_mediciones_por_dato)
    labels_tst = torch.tensor( [0]*len(sinPico_tst)+[1]*len(conPico_tst))#.view(11,1)
    
    trn_data = TensorDataset( data_trn, labels_trn)
    tst_data = TensorDataset( data_tst, labels_tst)
    
    B=100
    trn_load = DataLoader( trn_data, shuffle=True, batch_size=B)
    tst_load = DataLoader( tst_data, shuffle=True, batch_size=B)


#    model = CNN()
#    optim = torch.optim.Adam(model.parameters())
#    costf = torch.nn.CrossEntropyLoss()
#    costf = torch.nn.MSELoss()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    acc2=[]
    T_max=40
    paso=2
    
    start = time.time()
#    print("hello")
    for T in np.arange(paso,T_max+paso,paso):
#    T = 50
        model = CNN().to( device)
        optim = torch.optim.Adam(model.parameters())
        costf = torch.nn.CrossEntropyLoss()

        model.train()
        for t in range(T):
          E = 0
          for data, label in trn_load:
            data = data.to( device)
            label = label.to( device)
            optim.zero_grad()
            y = model(data)
            error = costf(y, label)
            error.backward()
            optim.step()
            E += error.item()
#          print(t, E) 
          
          
        model.eval()
        right, total = 0, 0
        with torch.no_grad():
            for data, labels in tst_load:
                data = data.to( device)
                labels = labels.to( device)
                y = model(data)
                right += (y.argmax(dim=1)==labels).sum().item()
                total += len(labels)
    
        accuracy = right / total
        acc2.append(accuracy)
        print('Accuracy: ', round(accuracy,2))
    
    end = time.time()
    print("Tiempo ejecución: ", end - start)

    plt.xlabel(u"Épocas")
    plt.ylabel("Accuracy en test")
    plt.plot( np.arange(paso,T_max+paso,paso), acc2)
    plt.show()
    
    
    

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
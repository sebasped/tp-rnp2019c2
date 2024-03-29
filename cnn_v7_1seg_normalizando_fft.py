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
import numpy as np
import time
import random


class CNN(torch.nn.Module):
    def __init__(self, in_chann=2, out_classes=2): 
        super().__init__()
        self.convLayers = torch.nn.Sequential(
                torch.nn.Conv1d(in_chann, out_channels=5, kernel_size=60, stride=1, padding=30, dilation=1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(5, 10, 30, 2, 30, dilation=1),
                torch.nn.MaxPool1d(2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(10, 20, 15, 2, 30, dilation=1),
                torch.nn.MaxPool1d(2),
                torch.nn.ReLU()
                )
        self.H = 20*29
        self.fcLayers = torch.nn.Sequential(
                torch.nn.Linear(self.H, 100),
                torch.nn.Tanh(),
#                torch.nn.Linear(300,100),
#                torch.nn.Tanh(),
                torch.nn.Linear(100,out_classes)
                )

    def forward(self, x):
        y1 = self.convLayers(x)
        y2 = y1.view( -1, self.H)
        y3 = self.fcLayers(y2)
        
        return y3


# Probar una lineal con clase única con activación sigmoidea y MSE

if __name__ == '__main__':
    
    sinPico=[]
    path = './'
    filename = 'sinPico1seg.txt'
    with open(path+filename, 'r') as f:
       lines = (line.strip() for line in f if line)
    #       xtest = [int(float(line)) for line in lines]
       for line in lines:
           floats = [float(x) for x in line.split()]
           sinPico.append(floats)
    
    conPico=[]
    path = './'
    filename = 'conPico1seg.txt'
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
    
    
    n=4
    # 80/20 train/test
    sinPico_trn = sinPico[:-N//n]
    sinPico_tst = sinPico[-N//n:]

    conPico_trn = conPico[:-N//n]
    conPico_tst = conPico[-N//n:]

    series_trn = sinPico_trn + conPico_trn
    series_tst = sinPico_tst + conPico_tst
    lt_trn = [0]*len(sinPico_trn)+[1]*len(conPico_trn)
    lt_tst = [0]*len(sinPico_tst)+[1]*len(conPico_tst)
    
    
    #por las dudas, mezclo los datos antes    
    entero_trn = list(zip(series_trn,lt_trn))
    random.shuffle(entero_trn)
    series_trn, lt_trn = zip(*entero_trn)

    entero_tst = list(zip(series_tst,lt_tst))
    random.shuffle(entero_tst)
    series_tst, lt_tst = zip(*entero_tst)
    
    
    # normalizo los datos de las series temporales
#    flattened_list = [y for x in series_trn+series_tst for y in x]
#    media = np.mean(flattened_list)
#    desvio = np.std(flattened_list)
#    normalizar = tv.transforms.Compose( [tv.transforms.Normalize(mean=[media], std=[desvio])] )
    
    cant_mediciones_por_dato = len( sinPico[0])
    data_trn_sinNorm = torch.tensor( series_trn)#.view(len(series_trn), 1, cant_mediciones_por_dato)
#    data_trn = normalizar(data_trn_sinNorm)
    data_trn = torch.rfft(data_trn_sinNorm,1,onesided=False).view(len(series_trn), 2, cant_mediciones_por_dato)
#    labels_trn = torch.tensor( [0]*len(sinPico_trn)+[1]*len(conPico_trn))#.view(11,1)
    labels_trn = torch.tensor( lt_trn)
    
    data_tst_sinNorm = torch.tensor( series_tst)#.view(len(series_tst), 1, cant_mediciones_por_dato)
#    data_tst = normalizar(data_tst_sinNorm)
    data_tst = torch.rfft(data_tst_sinNorm,1,onesided=False).view(len(series_tst), 2, cant_mediciones_por_dato)
#    labels_tst = torch.tensor( [0]*len(sinPico_tst)+[1]*len(conPico_tst))#.view(11,1)
    labels_tst = torch.tensor( lt_tst)

    
    trn_data = TensorDataset( data_trn, labels_trn)
    tst_data = TensorDataset( data_tst, labels_tst)
    
#    B=len(series_trn)
    B=1000
    trn_load = DataLoader( trn_data, shuffle=True, batch_size=B)
    tst_load = DataLoader( tst_data, shuffle=True, batch_size=B)


#    model = CNN()
#    optim = torch.optim.Adam(model.parameters())
#    costf = torch.nn.CrossEntropyLoss()
#    costf = torch.nn.MSELoss()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    acc2=[]
    T_max=50
    paso=5
    
    start = time.time()

    epocas = np.arange(paso,T_max+paso,paso)
    for T in epocas:
#    T = 50
        model = CNN().to( device)
        optim = torch.optim.Adam( model.parameters())
        costf = torch.nn.CrossEntropyLoss()

        model.train()
        for t in range(T):
          E = 0
          for data, label in trn_load:
            data = data.to( device)
            label = label.to( device)
            y = model(data)
            
            error = costf( y, label)
            error.backward()
            optim.step()
            optim.zero_grad()
            
            E += error.item()
#          print(t, E) 
        print('Error entrenamiento: ', round(E,4), 'Épocas: ', T)          
          
        model.eval()
        right, total = 0, 0
        with torch.no_grad():
            for data, labels in tst_load:
                data = data.to( device)
                labels = labels.to( device)
                y = model( data)
                right += ( y.argmax(dim=1)==labels).sum().item()
                total += len( labels)
    
        accuracy = right / total
        acc2.append(accuracy)
        print('Accuracy:', round(accuracy,3),'Épocas: ', T)
    
    end = time.time()
    print("Tiempo ejecución en minutos: ", round((end - start)/60,2) ) 

    print('Accuracy promedio', round(sum(acc2)/len(acc2),3) )
    plt.xlabel(u"Épocas")
    plt.ylabel("Accuracy en test")
    plt.ylim(0.8,1)
    plt.title('Promedio: %s -- Máximo: %s' %( round(sum(acc2)/len(acc2),3), round(max(acc2),3)) ) 
    plt.plot( epocas, acc2)
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
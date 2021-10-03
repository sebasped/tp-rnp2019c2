#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:18:28 2019

@author: sebas
"""

import torch
import torchvision as tv
from matplotlib import pyplot as plt


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
        y5 = y4.softmax(dim=1)

        return y5



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
    
    
    sinPico = sinPico[0:11]

    data = torch.tensor(sinPico)#.view(11,1,250) #o reshape
#    print('Data size:', data.shape)

    datafft = torch.rfft(data,1,onesided=True).view(11,2,126)
    print('Data size:', datafft.shape)
#    inv_normalize = tv.transforms.Compose(
#    [
#        tv.transforms.Normalize(mean=[0.5], std=[0.5])
#    ]
#    )
    
#    data_norm = inv_normalize(data)
    
#    c1 = torch.nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1)
#    y1 = c1(data)
#    print('y1 shape: ', y1.shape)


#    m = torch.nn.Conv1d(16, 33, 3, stride=2)
#    input = torch.randn(20, 16, 50)
#    output = m(input)
    

#    c2 = torch.nn.Conv1d(5, 10, 3, 1, 0)
#    y2 = c2(y1)
    
#    c2 = torch.nn.MaxPool1d(10)
#    y2 = c2(y1).relu()
#    print('y2 shape: ', y2.shape)
    
#    L = torch.nn.Linear(5*150,2)
#    y3 = L(y2.view(-1,5*150)).sigmoid()
#    print('y3 shape', y3.shape)
#    view(-1,32*8*8) #porque los Linears reciben vectores y no tensores
#    Linear(32*8*8, 512)
#    Linear(512, 10)    
    
    
    
    conv1 = torch.nn.Conv1d(2, out_channels=10, kernel_size=60, stride=1, padding=30)
    y1 = conv1(datafft).relu()
#    print(conv1)
    print('y1 shape: ', y1.shape)

    conv2 = torch.nn.Conv1d(10, 20, 30, 2, 30)
#    print(conv2)
    y2 = conv2(y1).relu()
    print('y2 shape: ', y2.shape)


    mpool1 = torch.nn.MaxPool1d(2)
#    print(mpool1)
    y3 = mpool1(y2).relu()
    print('y3 shape: ', y3.shape)
    
    conv3 = torch.nn.Conv1d(20, 40, 15, 2, 30)
#    print(conv3)
    y4 = conv3(y3).relu()
    print('y4 shape: ', y4.shape)

    mpool2 = torch.nn.MaxPool1d(2)
#    print(mpool2)
    y5 = mpool2(y4).relu()
    print('y5 shape: ', y5.shape)
    
    H = 40*21
    Linear1 = torch.nn.Linear(H, 200)
#    print(Linear1)
    y6 = Linear1(y5.view(-1,H)).tanh()
    print('y6 shape: ', y6.shape)
    
    Linear2 = torch.nn.Linear(200,2)
#    print(Linear2)
    y7 = Linear2(y6)
    print('y7 shape: ', y7.shape)
    
    
    
    
#    model = CNN()
#    ymodel=model(data)


    
#    model = CNN()
#    optim = torch.optim.Adam(model.parameters())
#    costf = torch.nn.CrossEntropyLoss()
#    
#    T = 20
#    model.train()
#    for t in range(T):
#      E = 0
#      for image, label in trn_load:
#        optim.zero_grad()
#        y = model(image)
#        error = costf(y, label)
#        error.backward()
#        optim.step()
#        E += error.item()
#      print(t, E) 
#      
#      
#    model.eval()
#    right, total = 0, 0
#    with torch.no_grad():
#        for images, labels in tst_load:
#            y = model(images)
#            right += (y.argmax(dim=1)==labels).sum().item()
#            total += len(labels)
#
#    accuracy = right / total
#    print('Accuracy: ', accuracy)
#
#
#
#"""
#puedo usar la red convolucional para hacer un autoencoder
#
#_.enc = torch.nn.Sequential(
#  conv2d(...),
#  torch.nn.ReLU(True))
#
#_.dec = torch.nn.ConvTranspose2d()
#Tanh()
#"""
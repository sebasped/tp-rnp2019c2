#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:18:28 2019

@author: sebas
"""

import torch
import torchvision as tv


class CNN(torch.nn.Module):
    def __init__(self, in_chann=1, out_classes=2): 
        super().__init__()
        self.c1 = torch.nn.Conv1d(in_chann, 16, 5, 1, 0)
#        self.c2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.c3 = torch.nn.MaxPool2d(2,2)
        self.H = 32*8*8
        self.L1 = torch.nn.Linear(self.H, 512)
        self.L2 = torch.nn.Linear(512,out_classes)

    def forward(self, x):
        y1 = self.c1(x).relu()
        y2 = self.c2(y1)
        y3 = self.c3(y2).relu()
        y4 = self.L1(y3.view(-1,self.H)).tanh()
        y5 = self.L2(y4)

        return y5


if __name__ == '__main__':
    
#    transf = tv.transforms.Compose( [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5],[0.5],[0.5]) ] )
#    B = 100
#    trn_data =tv.datasets.CIFAR10(root='./data', train = True, download = True, transform = transf) 
#    tst_data =tv.datasets.CIFAR10(root='./data', train = False, download = True, transform = transf)
#    trn_load =torch.utils.data.DataLoader(dataset = trn_data, batch_size = B, shuffle = True)
#    tst_load =torch.utils.data.DataLoader(dataset = tst_data, batch_size = B, shuffle = True)
    
    #idata = iter(trn_load)
    #image, label = next(idata)
    #print('image shape: ', image.shape)
    #print('label shape: ', label.shape)
    
    #c1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
    
    #y1 = c1(image)
    #print('c1 shape: ', y1.shape)
    
    #c2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
    #y2 = c2(y1)
    
    #c3 = torch.nn.MaxPool2d(2,2)
    #y3 = c3(y2)
    
    #view(-1,32*8*8) #porque los Linears reciben vectores y no tensores
    #Linear(32*8*8, 512)
    #Linear(512, 10)
    
    
    path = './'
    #filename = '0.1BzATP12.txt'
    filename = 'BEADSSignal.txt'
    with open(path+filename, 'r') as f:
        lines = (line.strip() for line in f if line)
        xorig = [float(line) for line in lines]

    # promedio para que no sea tan pesado.
    sig=[]
    N = 10  #El original es a 250 Hz. Queda a 250/N Hz.
    for j in range(len(xorig)//N):
        if j < (len(xorig)//N)-1:
            sig.append( sum(xorig[N*j:N*(j+1)]) /N )
    else:
        j=(len(xorig)//N)-1
        sig.append( sum(xorig[N*j:]) / len(xorig[N*j:]) )


    path = './'
    #filename = '0.1BzATP12.txt'
    filename = 'RedIntervalLines.txt'
    with open(path+filename, 'r') as f:
        lines = (line.strip() for line in f if line)
        redLines = [int(float(line)) for line in lines]


    redLinesConv = [i*250/N for i in redLines]
    redLinesConv.append(len(sig)-1)

    conPico=[]
    sinPico=[]

    sizeWinSegs = 50
    sizeWin = int(sizeWinSegs*250/N)

    sin=True
    for i in range(len(redLinesConv)):
        j=0
        if i==0:
            offset=0
        else:
            offset = int(redLinesConv[i-1])
        if (sin==True):
            while (j+1)*sizeWin+offset < redLinesConv[i]:
                sinPico.append( sig[j*sizeWin+offset:(j+1)*sizeWin+offset] )
                j += 1
                sin=False
        else:
            j=0
            while (j+1)*sizeWin+offset < redLinesConv[i]:
                conPico.append( sig[j*sizeWin+offset:(j+1)*sizeWin+offset] )
                j += 1
            sin=True

    
    
#    raw = conPico[0]
    data = torch.tensor(conPico).view(11,1,1250) #o reshape
    print('Data size:', data.shape)
    c1 = torch.nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=0)
    y1 = c1(data)
    print('y1 shape: ', y1.shape)


#    m = torch.nn.Conv1d(16, 33, 3, stride=2)
#    input = torch.randn(20, 16, 50)
#    output = m(input)
    


#    c2 = torch.nn.Conv1d(5, 10, 3, 1, 0)
#    y2 = c2(y1)
    
    c2 = torch.nn.MaxPool1d(2,2)
    y2 = c2(y1).relu()
    print('y2 shape: ', y2.shape)
    
    L = torch.nn.Linear(5*624,2)
    y3 = L(y2.view(-1,5*624)).softmax(dim=1)
    print('y3 shape', y3.shape)
#    view(-1,32*8*8) #porque los Linears reciben vectores y no tensores
#    Linear(32*8*8, 512)
#    Linear(512, 10)    
    




    
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
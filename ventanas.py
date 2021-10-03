#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:43:37 2019

@author: sebas
"""

from matplotlib import pyplot as plt
#import numpy as np

path = './'
#filename = '0.1BzATP12.txt'
filename = 'BEADSSignal.txt'
with open(path+filename, 'r') as f:
       lines = (line.strip() for line in f if line)
       xorigBeads = [float(line) for line in lines]

path = './'
filename = '0.1BzATP12.txt'
#filename = 'OriginalSignalSinT.txt'
with open(path+filename, 'r') as f:
       lines = (line.strip() for line in f if line)
       xorig = [float(line) for line in lines]


x=[]
path = './'
##filename = '0.1BzATP12.txt'
filename = 'sinPico.txt'
with open(path+filename, 'r') as f:
       lines = (line.strip() for line in f if line)
#       xtest = [int(float(line)) for line in lines]
#       line=f.readline()
       for line in lines:
           floats = [float(x) for x in line.split()]
           x.append(floats)


plt.clf()
plt.plot(xorig[:30000], label=u'Señal original')
#plt.plot(xorigBeads[:30000], label=u'Señal corregida')
plt.legend()
plt.xlabel('Mediciones a 250 Hz')
plt.ylabel('Intensidad señal')

plt.show()




# promedio para que no sea tan pesado.
sig=[]
N = 10  #El original es a 250 Hz. Queda a 250/N Hz.
for j in range(len(xorig)//N):
    if j < (len(xorig)//N)-1:
        sig.append( sum(xorig[N*j:N*(j+1)]) /N )
    else:
        j=(len(xorig)//N)-1
        sig.append( sum(xorig[N*j:]) / len(xorig[N*j:]) )


plt.clf()
#plt.plot(xorig)
plt.plot(sig)
#plt.plot(xtest[0])
#plt.show()


path = './'
#filename = '0.1BzATP12.txt'
filename = 'RedIntervalLines.txt'
with open(path+filename, 'r') as f:
       lines = (line.strip() for line in f if line)
       redLines = [float(line) for line in lines]


redLinesConv = [i*250/N for i in redLines]
#redLinesConv.append(15600)
#redLinesConv[1]=10000
redLinesConv.append(len(sig)-1)


for i in range(len(redLinesConv)):
    plt.axvline(redLinesConv[i], color='r')
plt.axvline(redLinesConv[1], color='r')
plt.show()


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
#        print(j)
        while (j+1)*sizeWin+offset < redLinesConv[i]:
#            print('entre')
            conPico.append( sig[j*sizeWin+offset:(j+1)*sizeWin+offset] )
            j += 1
        sin=True



#sinPico.append( sig[j*sizeWin:(j+1)*sizeWin] )


#plt.plot(conPico[0])

#array = np.array(conPico)
#np.savetxt('test.txt', conPico)
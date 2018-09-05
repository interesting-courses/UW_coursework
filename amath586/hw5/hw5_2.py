#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve1d

#<start>
a = 1
h = 0.05
k = 0.8*h
T = 17

x = np.arange(0,25+h,h)
t = np.arange(0,T+k,k)

u = np.zeros((len(t),len(x)))

u[0] = np.exp(-20*(x-2)**2) + np.exp(-(x-5)**2)
u[1] = u0 - k*a*(convolve1d(u0,[1,0,-1],mode='constant'))/(2*h)

for n in range(1,len(t)-1):
    u[n+1] = convolve1d(u[n-1],[0,0,0,0,1],mode='constant') - (a*k/h - 1) * (u[n] - convolve1d(u[n],[0,0,0,0,1],mode='constant'))
#<end>
    
mpl.rcParams['text.usetex'] = True
xs = x[int((len(x)-1)*3/5):]
us = np.exp(-20*(xs-2-17)**2) + np.exp(-(xs-5-17)**2)
plt.figure
plt.plot(xs,us,color='0.75')
plt.plot(xs,u[-1,int((len(x)-1)*3/5):],color='k',marker='.',linestyle='None')
plt.xticks([15,20,25])
plt.yticks([-0.5,0,.5,1,1.5])
plt.savefig('wave.pdf')
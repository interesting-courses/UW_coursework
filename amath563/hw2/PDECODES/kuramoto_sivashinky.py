#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Kuramoto-Sivashinsky equation (from Trefethen)
# u_t = -u*u_x - u_xx - u_xxxx,  periodic BCs 

N = 1024;
x = 32*np.pi*np.arange(1,N+1)/N;
u = np.cos(x/16)*(1+np.sin(x/16)); 
v = np.fft.fft(u);

#Spatial grid and initial condition:
h = 0.025;
k = np.concatenate([np.arange(0,N/2),[0],np.arange(-N/2+1,0)])/16;
L = k**2 - k**4;
E = np.exp(h*L); E2 = np.exp(h*L/2);
M = 16;
r = np.exp(1j*np.pi*(np.arange(1,M+1)-.5)/M)
LR = h*np.tile(L,(M,1)).T + np.tile(r,(N,1))
Q = h*np.real(np.mean( (np.exp(LR/2)-1)/LR, axis=1)); 
f1 = h*np.real(np.mean( (-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1)); 
f2 = h*np.real(np.mean( (2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1));
f3 = h*np.real(np.mean( (-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1));

# Main time-stepping loop:
uu = u; tt = 0;
tmax = 100; nmax = round(tmax/h); nplt = np.floor((tmax/250)/h); g = -0.5j*k;
for n in range(1,nmax+1):
    t = n*h;
    Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2);
    a = E2*v + Q*Nv;
    Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2);
    b = E2*v + Q*Na;
    Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2);
    c = E2*a + Q*(2*Nb-Nv);
    Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2);
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3; 
    if np.mod(n,nplt)==0:
        u = np.real(np.fft.ifft(v));
        uu = np.vstack([uu,u]); tt = np.append(tt,t);

# Plot results:
plt.figure()
plt.pcolormesh(uu)
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse import csgraph

#%%
def kura_rhs(t,theta,omega,n,K,A):
    coupling = np.zeros(n)
    for j in range(n):
        C = 0;
        for i in range(n):
            C = C+A[i,j]*np.sin(theta[i]-theta[j])
        coupling[j] = C
    
    return omega + (K/n)*coupling

#%%

T=100
t=np.linspace(0,T,2001);

K=2;  # coupling strength
n=10; # number of oscillators
rad=np.ones((n,1));
thetai=2*np.random.randn(n);
omega=np.random.rand(n)+0.5;

A=np.random.rand(n,n);  
A[np.where(A<=0.5)] = 0;

sol = sp.integrate.solve_ivp(lambda t,theta:kura_rhs(t,theta,omega,n,K,A),(0,T),thetai)

t = sol.t
theta = sol.y

#%%

plt.figure()
plt.plot(t,theta.T)
plt.show()
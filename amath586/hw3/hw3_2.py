#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)
from scipy import integrate
from scipy import io

#<start:mp>
def mp(f,u0,u1,N,k):
    u = np.zeros(N)
    u[0] = u0
    u[1] = u1
    for n in np.arange(1,N-1):
        u[n+1] = u[n-1] + 2*k*f(u[n])
    return u
#<end:mp>
    y
eta = 1
alpha = -10

def f(u):
    return alpha * u

T = 1
for N in [100,1000,3000,10000]:
    k = T/N
    
    u0 = eta
    u1 = eta*(1+k*alpha)
    
    u = mp(f,u0,u1,N,k)

    t = np.linspace(0,1,N)
    u_true = eta * np.exp(alpha * t)

    plt.figure()
    plt.plot(t,u_true,color='0.8', linewidth=6)
    plt.plot(t,u,color='0')
    plt.savefig('img/mp_'+str(N)+'.pdf')
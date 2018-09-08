#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)

from data_driven_modeling import diff,diff2,lasso

from scipy import io
burgers_mat = io.loadmat('burgers.mat')

#%%
    
x = burgers_mat['x'][0]
t = burgers_mat['t'][:,0]
usol = burgers_mat['usol']

u = np.real(usol)
[m,n] = np.shape(u)

dt = t[1] - t[0];
u_t = diff(u,dt,axis=1,endpoints=True);


# %%
'''
D = np.diag(np.ones(m-1),1)-np.diag(np.ones(m-1),-1)
D[0,m-1] = -1
D[m-1,0] = 1
D /= 2*dx


D2 = -2*np.diag(np.ones(m))+np.diag(np.ones(m-1),1)+np.diag(np.ones(m-1),-1)
D2 /= (dx)**2
D2[m-1,0]=1;
D2[0,m-1]=1;

ux = D@X[:,1:-1]
uxx = D2@X[:,1:-1]
u2x = D@(X[:,1:-1]**2)

u=np.reshape( X[:,1:-1].T,(n-2)*m,1)
Ux = np.reshape(ux.T,(n-2)*m,1)
Uxx = np.reshape(uxx.T,(n-2)*m,1)
U2x = np.reshape(u2x.T,(n-2)*m,1)
'''
#%%

dx = x[1] - x[0];
ux = diff(u,dx,endpoints=True)
uxx = diff2(u,dx,endpoints=True)
u2x = diff(u**2,dx,endpoints=True)
A = np.array([u,u**2,u**3, ux, uxx, u2x, ux*u, ux*ux, ux*uxx]).reshape(-1,m*n).T

#%%
'''
U=np.reshape(u,-1);
Ux = np.reshape(ux,-1);
Uxx = np.reshape(uxx,-1);
U2x = np.reshape(u2x,-1);
U_t=np.reshape(u_t,-1);


AA=np.array([U, U**2, U**3, Ux, Uxx, U2x, Ux*U, Ux*Ux, Ux*Uxx]).T;
'''


#%%
xi= np.linalg.lstsq(A,u_t.reshape(-1))[0];

#%%
xi = lasso(A,u_t.reshape(-1),alpha=0.02)

#%%
plt.figure();
plt.bar(np.arange(1,len(xi)+1),xi);
plt.show();


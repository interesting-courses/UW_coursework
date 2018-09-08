#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#<start>
import numpy as np
from scipy import io,integrate
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)

from scipy.interpolate import interp1d
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import gaussian_filter

from data_driven_modeling import get_poly_exponents,bin_exp,diff,pdiff,diff2,lasso, kl_divergence,AIC,BIC


BZ_tensor=np.load('BZ.npz')['BZ_tensor'].astype('int16')

#%% pick interesting smaller region
region = (slice(80,210),slice(40,170),slice(650,None))

I = BZ_tensor[region].astype('float64')

#%% save plots of image

plt.figure()
t = 650
m = plt.pcolormesh(BZ_tensor[:,:,t])
m.set_rasterized(True)
plt.plot([40,170],[80,210], color='white', linestyle=':')
plt.plot([340,440],[320,220], color='white', linestyle='--')
plt.plot([40,40,170,170,40],[80,210,210,80,80], color='white')
plt.axis('image')
plt.savefig('img/BZ/slice_'+str(t)+'.pdf')


#%% TRY 1D SLICES

u = I[np.arange(130),np.arange(130)]
#u = BZ_tensor[np.arange(320,220,-1),np.arange(340,440),400:]

[M,T] = np.shape(u)

plt.figure()
plt.scatter(np.arange(M),u[:,200])
plt.show()

dt = 1;
u_t = diff(u,dt,axis=1,endpoints=True);
u_tt = diff2(u,dt,axis=1,endpoints=True);

dx = 1;
u_x = diff(u,dx,endpoints=True)
u_xx = diff2(u,dx,endpoints=True)
u2_x = diff(u**2,dx,endpoints=True)

A = np.array([u,u**2,u**3, u_x, u_xx, u_x*u, u_x*u_x, u_x*u_xx]).reshape(-1,M*T).T

#%% regress

xi= np.linalg.lstsq(A,u_t.reshape(-1))[0];

#%%
plt.figure();
plt.bar(np.arange(1,len(xi)+1),xi);
plt.show();


#%% define RHS for differential equation
def f(t,y):    
    dx = 1;
    y_x = diff(y,dx,endpoints=True)
    y_xx = diff2(y,dx,endpoints=True)
    y2_x = diff(y**2,dx,endpoints=True)
    
    return np.array([y,y**2,y**3, y_x, y_xx, y_x*y, y_x*y_x, y_x*y_xx]).reshape(-1,len(y)).T @ xi

#%% run forward euler to solve

ys = np.zeros((M,T))
dt = 1
ys[:,0] = u[:,0]

for i in range(T-1):
    ys[:,i+1] = ys[:,i] + dt * f(0,ys[:,i])

#%% solve using Runge-Kutta 45

sol = integrate.solve_ivp(f,[0,550],u[40],method='RK45')

plt.scatter(sol.t, sol.y[1])

#%% TRY 2d cut

# what is sensor frequency?
[M,N,T]=np.shape(I)

dt = 1
I_t = diff(I,dt,axis=2,endpoints=True)
I_tt = diff2(I,dt,axis=2,endpoints=True)


# build library 
dx = 1
dy = 1

I_x = diff(I,dx,axis=0,endpoints=True)
I_y = diff(I,dy,axis=1,endpoints=True)
I_xx = diff2(I,dx,axis=0,endpoints=True)
I_xy = pdiff(I,[dx,dy],axis=[0,1],endpoints=True)
I_yy = diff2(I,dy,axis=1,endpoints=True)

lib = np.array([I, I_x, I_y, I_xx, I_xy, I_yy]).reshape(-1,M*N*T).T;
lib2 = np.array([I, I_x, I_y, I_xx, I_xy, I_yy ]).reshape(-1,M*N*T).T;

#%% compute coefficients
c = np.linalg.lstsq(lib,I_t.reshape(-1))[0]
c2 = np.linalg.lstsq(lib2,I_tt.reshape(-1))[0]

#%% helper function for AIC BIC
def lib_at_c(I,c):
    dx = 1
    dy = 1
    
    I_x = diff(I,dx,axis=0,endpoints=True)
    I_y = diff(I,dy,axis=1,endpoints=True)
    I_xx = diff2(I,dx,axis=0,endpoints=True)
    I_xy = pdiff(I,[dx,dy],axis=[0,1],endpoints=True)
    I_yy = diff2(I,dy,axis=1,endpoints=True)
    
    return (np.array([I, I_x, I_y, I_xx, I_xy, I_yy]).reshape(-1,M*N).T @ c);

#%% compute library at all times
libs = np.zeros((M*N,T))
libs2 = np.zeros((M*N,T))
for t in range(T):
    libs[:,t] = lib_at_c(I[:,:,t],c)
    libs2[:,t] = lib_at_c(I[:,:,t],c2)

#%% compute AIC BIC
    
data = np.array([[AIC(libs,I_t.reshape(M*N,T),len(c)), BIC(libs,I_t.reshape(M*N,T),len(c))]])
data2 = np.array([[AIC(libs2,I_t.reshape(M*N,T),len(c)), BIC(libs2,I_t.reshape(M*N,T),len(c))]])

np.savetxt('img/BZ/2d.txt',data, fmt='%.2f, %.2f')
np.savetxt('img/BZ/2d2.txt',data2, fmt='%.2f, %.2f')


#%% plot coefficients

plt.figure();
plt.bar(np.arange(1,len(c)+1),c);
plt.show();

plt.figure();
plt.bar(np.arange(1,len(c2)+1),c2);
plt.show();

#%% define RHS for sovler

def f(t,y):
    dx = 1
    dy = 1
    
    I = np.reshape(y,(M,N))
    
    I_x = diff(I,dx,axis=0,endpoints=True)
    I_y = diff(I,dx,axis=1,endpoints=True)
    I_xx = diff2(I,dx,axis=0,endpoints=True)
    I_xy = pdiff(I,[dx,dx],axis=[0,1],endpoints=True)
    I_yy = diff2(I,dx,axis=1,endpoints=True)
    
    return np.array([I, I_x, I_y, I_xx, I_xy, I_yy]).reshape(-1,M*N).T @ c;

#%% solve
sol = integrate.solve_ivp(f,[0,T],np.reshape(I[:,:,0],-1),t_eval = np.arange(T))
ts = sol.t
ys = np.reshape(sol.y,(M,N,len(ts)))

#%% define rhs for forward euler and integrate vs time twice

def f2(I):
    dx = 1
    dy = 1
        
    I_x = diff(I,dx,axis=0,endpoints=True)
    I_y = diff(I,dx,axis=1,endpoints=True)
    I_xx = diff2(I,dx,axis=0,endpoints=True)
    I_xy = pdiff(I,[dx,dx],axis=[0,1],endpoints=True)
    I_yy = diff2(I,dx,axis=1,endpoints=True)
    
    return (np.array([I, I_x, I_y, I_xx, I_xy, I_yy]).reshape(-1,M*N).T @ c2).reshape(M,N);  

ys = np.zeros((M,N,T))

ys[:,:,[0,1]] = I[:,:,[0,1]]

dt = .5

for i in range(1,500):
    ys[:,:,i+1] = 2*ys[:,:,i] - ys[:,:,i-1] + dt * f2(ys[:,:,i])


#%% 

plt.figure()
plt.pcolormesh(ys[:,:,300])
plt.axis('image')
#<end>
#%%
#for j in np.arange(10):
'''
j=450
A=I[:,:,j];

#%%
from matplotlib import animation

#%%
def animate(i):
    plt.pcolormesh(I[:,:,i])

#%matplotlib qt5
plt.rc('animation', html='html5')

fig = plt.figure()

plt.pcolormesh(I[:,:,0])
plt.axis('image')

anim = animation.FuncAnimation(fig, animate, frames = np.arange(2,500,10), blit = False)

plt.draw()
plt.show()


    
#%%
plt.figure(I[:,:,10])
plt.pcolormesh(A)
plt.axis('image')
plt.show()

plt.figure()
blurred = maximum_filter(A,footprint=np.ones((5,5)))
plt.pcolormesh(blurred)
plt.axis('image')
plt.show()

footprint = np.ones((6,6));
footprint[0,0] = 0
footprint[0,5] = 0
footprint[5,0] = 0
footprint[5,5] = 0

plt.figure()
blurred = maximum_filter(A,footprint=footprint)
plt.pcolormesh(blurred)
plt.axis('image')
plt.show()

'''
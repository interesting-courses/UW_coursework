#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 08:07:06 2018

@author: tyler
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
  solve SDE dX_t = a(t,X_t,t)dt + b(t,X_t)dW_t 

    Parameters
    ----------
    mu : callable mu(t,X_t),
        t is scalar time and X_t is vector position
    sigma : callable sigma(t,X_t),
        where t is scalar time and X_t is vector position
    x0 : ndarray
        the initial position
    t : ndarray
        list of times at which to evaluate trajectory

    Returns
    -------
    x : ndarray
        positions of trajectory at each time in t
    dWt : incriments of Brownian Motion used
'''
#<start:em>
def euler_maruyama(a,b,x0,t,dWt):
    N = len(t)
    x = np.zeros((N,len(x0)))
    x[0] = x0
    for i in range(N-1):
        dt = t[i+1]-t[i]
        x[i+1] = x[i] + a(t[i],x[i])*dt + b(t[i],x[i])*dWt[i+1]
    return x
#<end:em>
#<start:rk>
def runge_kutta(a,b,x0,t,dWt):
    N = len(t)
    x = np.zeros((N,len(x0)))
    x[0] = x0
    for i in range(N-1):
        dt = t[i+1]-t[i]
        x_hat = x[i]+b(t[i],x[i])*np.sqrt(dt)
        x[i+1] = x[i] + a(t[i],x[i])*dt + b(t[i],x[i])*dWt[i+1] 
                   + (b(t[i],x_hat)-b(t[i],x[i]))*(dWt[i+1]**2-dt)/(2*np.sqrt(dt))
    return x
#<end:rk>

#%% OU Process
    
theta = 10
u = 1
s = 0.5

def mu(t,x):
    return theta*(u-x)

def sigma(t,x):
    return s

#%% Plot some trajectories
N = 251
T = 1
t = np.linspace(0,T,N)

x0 = [0]

num_trajectories = 10

plt.figure()
for i in range(num_trajectories):
    dWt = np.random.normal(0,1,size=(N))*np.sqrt(t[1])
    x_em = euler_maruyama(mu,sigma,x0,t,dWt)
    plt.plot(t,x_em,alpha=1/num_trajectories,color='k')
plt.savefig('img/OU_em_'+str(num_trajectories)+'.pdf')

plt.figure()
for i in range(num_trajectories):
    dWt = (-1+2*np.random.randint(2,size=(N)))*np.sqrt(t[1])
    x_em2 = euler_maruyama(mu,sigma,x0,t,dWt)
    plt.plot(t,x_em2,alpha=1/num_trajectories,color='k')
plt.savefig('img/OU_em_2_'+str(num_trajectories)+'.pdf')

plt.figure()
for i in range(num_trajectories):
    dWt = np.random.normal(0,1,size=(N))*np.sqrt(t[1])
    x_rk = runge_kutta(mu,sigma,x0,t,dWt)
    plt.plot(t,x_rk,alpha=1/num_trajectories,color='k')
plt.savefig('img/OU_rk_'+str(num_trajectories)+'.pdf')


#%%

theta = 2
u = 1
s = 1/100

def mu(t,x):
    return theta*(u-x)

def sigma(t,x):
    return s

Ns = 2**np.linspace(3,9,7)
#Ns = 2**np.linspace(2,7,6)
#Ns = 2**np.linspace(4,9,6)
#Ns = 2**np.linspace(5,12,4)
#Ns = 2**np.linspace(4,10,7)

expected_error_em = np.zeros(len(Ns))
expected_error_em2 = np.zeros(len(Ns))
expected_error_rk = np.zeros(len(Ns))

num_trajectories= 2000
T = 1

x_T_em = np.zeros((num_trajectories,len(Ns)))
x_T_em2 = np.zeros((num_trajectories,len(Ns)))
x_T_rk = np.zeros((num_trajectories,len(Ns)))

x0 = np.array([2])

for k,N in enumerate(Ns):
    print(N)
    t = np.linspace(0,T,N)
    
    

    for i in range(num_trajectories):
        dWt = np.random.normal(0,1,size=(int(N)))*np.sqrt(t[1])
        dWt[0] = 0
        
        x_em = euler_maruyama(mu,sigma,x0,t,dWt)
        x_T_em[i,k] = x_em[-1]
        
        x_rk = runge_kutta(mu,sigma,x0,t,dWt)
        x_T_rk[i,k] = x_rk[-1]
        
        dWt_discrete = (-1+2*np.random.randint(2,size=(int(N))))*np.sqrt(t[1])
        dWt_discrete[0] = 0
        
        x_em2 = euler_maruyama(mu,sigma,x0,t,dWt_discrete)
        x_T_em2[i,k] = x_em2[-1]

#%% First Moment

mean_x_T_true = (x0 - u)*np.exp(-theta*T)+u

mean_error_em = np.abs(np.mean(x_T_em,axis=0) - mean_x_T_true)/mean_x_T_true
mean_error_em2 = np.abs(np.mean(x_T_em2,axis=0) - mean_x_T_true)/mean_x_T_true
mean_error_rk = np.abs(np.mean(x_T_rk,axis=0) - mean_x_T_true)/mean_x_T_true


linear_fit_em = np.poly1d(np.polyfit(np.log10(T/Ns), np.log10(mean_error_em), 1))
linear_fit_em2 = np.poly1d(np.polyfit(np.log10(T/Ns), np.log10(mean_error_em2), 1))
linear_fit_rk = np.poly1d(np.polyfit(np.log10(T/Ns), np.log10(mean_error_rk), 1))

print(linear_fit_em)
print(linear_fit_em2)
print(linear_fit_rk)

plt.figure()
plt.yscale('log')
plt.xscale('log')

plt.plot(T/Ns,10**linear_fit_em(np.log10(T/Ns)), color='0.8')
plt.plot(T/Ns,10**linear_fit_em2(np.log10(T/Ns)), color='0.8')
plt.plot(T/Ns,10**linear_fit_rk(np.log10(T/Ns)), color='0.8')

plt.plot(T/Ns,mean_error_em, color='0', marker='o', Linestyle='None')
plt.plot(T/Ns,mean_error_em2, color='0', marker='s', Linestyle='None')
plt.plot(T/Ns,mean_error_rk, color='0', marker='^', Linestyle='None')

#%%
plt.savefig('img/weak_order_'+str(num_trajectories)+'.pdf')

np.savetxt('img/weak_order_em_'+str(num_trajectories)+'.txt',[linear_fit_em[1]],fmt='%.4f')
np.savetxt('img/weak_order_em2_'+str(num_trajectories)+'.txt',[linear_fit_em[1]],fmt='%.4f')
np.savetxt('img/weak_order_rk_'+str(num_trajectories)+'.txt',[linear_fit_rk[1]],fmt='%.4f')


#%% p-th Moment E[(x-E[x])^p]

p = 2
pmom_x_T_true = (s**p/(p*theta))*(1 - np.exp(-p*theta*T))

pmom_error_em = np.abs(np.mean((x_T_em-mean_x_T_true)**p,axis=0) - pmom_x_T_true)/pmom_x_T_true
pmom_error_em2 = np.abs(np.mean((x_T_em2-mean_x_T_true)**p,axis=0) - pmom_x_T_true)/pmom_x_T_true
pmom_error_rk = np.abs(np.mean((x_T_rk-mean_x_T_true)**p,axis=0) - pmom_x_T_true)/pmom_x_T_true


linear_fit_em = np.poly1d(np.polyfit(np.log10(T/Ns), np.log10(pmom_error_em), 1))
linear_fit_em2 = np.poly1d(np.polyfit(np.log10(T/Ns), np.log10(pmom_error_em2), 1))
linear_fit_rk = np.poly1d(np.polyfit(np.log10(T/Ns), np.log10(pmom_error_rk), 1))

print(linear_fit_em)
print(linear_fit_em2)
print(linear_fit_rk)

plt.figure()
plt.yscale('log')
plt.xscale('log')

plt.plot(T/Ns,10**linear_fit_em(np.log10(T/Ns)), color='0.8')
plt.plot(T/Ns,10**linear_fit_em2(np.log10(T/Ns)), color='0.8')
plt.plot(T/Ns,10**linear_fit_rk(np.log10(T/Ns)), color='0.8')

plt.plot(T/Ns,pmom_error_em, color='0', marker='o', Linestyle='None')
plt.plot(T/Ns,pmom_error_em2, color='0', marker='s', Linestyle='None')
plt.plot(T/Ns,pmom_error_rk, color='0', marker='^', Linestyle='None')

#%%
plt.savefig('img/weak_order_'+str(p)+'_mom_'+str(num_trajectories)+'.pdf')

np.savetxt('img/weak_order_em_'+str(p)+'_mom_'+str(num_trajectories)+'.txt',[linear_fit_em[1]],fmt='%.4f')
np.savetxt('img/weak_order_em2_'+str(p)+'_mom_'+str(num_trajectories)+'.txt',[linear_fit_em[1]],fmt='%.4f')
np.savetxt('img/weak_order_rk_'+str(p)+'_mom_'+str(num_trajectories)+'.txt',[linear_fit_rk[1]],fmt='%.4f')


#%% Geometric Brownian Motion
u = 1
s = 0.5

def mu(t,x):
    return u*x

def sigma(t,x):
    return s*x

def x_true_(x0,t,Wt):
    return x0*np.exp((u-s**2/2)*t+s*Wt)

#%% Plot some sample trajectories of Geometric Brownian Motion

N = 201
T = 1
t = np.linspace(0,T,N)

dWt = np.random.normal(0,1,size=(N))*np.sqrt(t[1])
dWt[0] = 0

x0 = [300]

x_true = x_true_(x0,t,np.cumsum(dWt))

x_em = euler_maruyama(mu,sigma,x0,t[::20],np.append([0],np.diff(np.cumsum(dWt)[::20])))

print(x_true[-1]-x_em[-1])

plt.figure()
plt.plot(t,x_true,color='.7')
plt.plot(t[::20],x_em,'o',Linestyle='None',color='k')
plt.savefig('img/GBM_true_vs_10.pdf')

x_rk = runge_kutta(mu,sigma,x0,t,dWt)

print(x_true[-1]-x_rk[-1])

plt.figure()
plt.plot(t,x_rk)
plt.plot(t,x_true)
plt.show()



#%%    

Ns = 2**np.linspace(5,12,8)
expected_error_em = np.zeros(len(Ns))
expected_error_rk = np.zeros(len(Ns))

num_trajectories= 1000
T = 1

for k,N in enumerate(Ns):
    print(N)
    t = np.linspace(0,T,N)
    x0 = [300]
    
    error_em = np.zeros(num_trajectories)
    error_rk = np.zeros(num_trajectories)
    
    for i in range(num_trajectories):
        dWt = np.random.normal(0,1,size=(int(N)))*np.sqrt(t[1])
        dWt[0] = 0
        
        x_true = x_true_(x0,t,np.cumsum(dWt))

        x_em = euler_maruyama(mu,sigma,x0,t,dWt)
        error_em[i] = np.abs(x_true[-1] - x_em[-1])
        
        x_rk = runge_kutta(mu,sigma,x0,t,dWt)
        error_rk[i] = np.abs(x_true[-1] - x_rk[-1]) 
    
    expected_error_em[k] = np.mean(error_em)
    expected_error_rk[k] = np.mean(error_rk)
     
    plt.figure()
    plt.hist(error_em)
    plt.show()

    plt.figure()
    plt.hist(error_rk)
    plt.show()
    
#%%

expected_error_em /= x0[0]*np.exp(u*T)
expected_error_rk /= x0[0]*np.exp(u*T)
    
linear_fit_em = np.poly1d(np.polyfit(np.log10(T/Ns), np.log10(expected_error_em), 1))
linear_fit_rk = np.poly1d(np.polyfit(np.log10(T/Ns), np.log10(expected_error_rk), 1))

print(linear_fit_em)
print(linear_fit_rk)

plt.figure()
plt.yscale('log')
plt.xscale('log')

plt.plot(T/Ns,10**linear_fit_em(np.log10(T/Ns)), color='0.8')
plt.plot(T/Ns,10**linear_fit_rk(np.log10(T/Ns)), color='0.8')

plt.plot(T/Ns,expected_error_em, color='0', marker='o', Linestyle='None')
plt.plot(T/Ns,expected_error_rk, color='0', marker='^', Linestyle='None')



plt.savefig('img/strong_order_'+str(num_trajectories)+'.pdf')

np.savetxt('img/strong_order_em_'+str(num_trajectories)+'.txt',[linear_fit_em[1]],fmt='%.4f')
np.savetxt('img/strong_order_rk_'+str(num_trajectories)+'.txt',[linear_fit_rk[1]],fmt='%.4f')
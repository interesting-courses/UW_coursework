#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:41:35 2018

@author: tyler
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse import csgraph


prams = sp.io.loadmat('prams.mat')

L = prams['L']
Ac = prams['Ac']

#%% set up weighted adjacency graphs
L_g = np.diag(np.diag(L)) - L
Ac_g = Ac


#%% compute degree distribution
L_deg = np.count_nonzero(L_g,axis=0)

Ac_deg_in = np.count_nonzero(Ac_g,axis=0)
Ac_deg_out = np.count_nonzero(Ac_g,axis=1)

plt.figure()
plt.hist(L_deg,np.arange(np.max(55)),color='k')
plt.savefig('img/L_deg.pdf')

plt.figure()
plt.hist(Ac_deg_in,np.arange(np.max(55)),color='k')
plt.savefig('img/Ac_deg_in.pdf')

plt.figure()
plt.hist(Ac_deg_out,np.arange(np.max(55)),color='k')
plt.savefig('img/Ac_deg_out.pdf')

#%% compute flow distribution
L_flow = np.sum(L_g,axis=0)

Ac_flow_in = np.sum(Ac_g,axis=0)
Ac_flow_out = np.sum(Ac_g,axis=1)

plt.figure()
plt.hist(L_flow,np.arange(np.max(245),step=2),color='k')
plt.savefig('img/L_flow.pdf')

plt.figure()
plt.hist(Ac_flow_in,np.arange(np.max(245),step=2),color='k')
plt.savefig('img/Ac_flow_in.pdf')

plt.figure()
plt.hist(Ac_flow_out,np.arange(np.max(245),step=2),color='k')
plt.savefig('img/Ac_flow_out.pdf')

#%% set up unweighted adjacency graphs and compute # connected components / diameters

L_adj = L_g!=0
L_components = sp.sparse.csgraph.connected_components(L_adj,directed=False)

Ac_adj = Ac_g!=0
Ac_strong_components = sp.sparse.csgraph.connected_components(Ac_adj,directed=True,connection='strong')
Ac_weak_components = sp.sparse.csgraph.connected_components(Ac_adj,directed=True,connection='weak')
 
Ac_weak_dists = sp.sparse.csgraph.dijkstra(Ac_adj+Ac_adj.T)
Ac_weak_diameter = np.max(Ac_weak_dists)

# construct directed adjacency graph of everything
C_g = L_g + Ac_g
C_adj = L_adj + Ac_adj
C_strong_components = sp.sparse.csgraph.connected_components(C_adj,directed=True,connection='strong')
C_weak_components = sp.sparse.csgraph.connected_components(C_adj,directed=True,connection='weak')

C_weak_dists = sp.sparse.csgraph.dijkstra(C_adj+C_adj.T)
C_weak_diameter = np.max(C_weak_dists)


#%% compute diameters of communication classes

L_diameters = np.zeros(L_components[0])
L_sizes = np.zeros(L_components[0])
for k in range(L_components[0]):
    rc = L_components[1]==k
    subgraph = L_adj[rc][:,rc]
    L_sizes[k] = len(subgraph)
    L_diameters[k] = np.max(sp.sparse.csgraph.dijkstra(subgraph))
    
Ac_diameters = np.zeros(Ac_strong_components[0])
Ac_sizes = np.zeros(Ac_strong_components[0])
for k in range(Ac_strong_components[0]):
    rc = Ac_strong_components[1]==k
    subgraph = Ac_adj[rc][:,rc]
    Ac_sizes[k] = len(subgraph)
    Ac_diameters[k] = np.max(sp.sparse.csgraph.dijkstra(subgraph))

C_diameters = np.zeros(Ac_strong_components[0])
C_sizes = np.zeros(Ac_strong_components[0])
for k in range(C_strong_components[0]):
    rc = C_strong_components[1]==k
    subgraph = C_adj[rc][:,rc]
    C_sizes[k] = len(subgraph)
    C_diameters[k] = np.max(sp.sparse.csgraph.dijkstra(subgraph))


#%%
def kura_rhs(t,theta,omega,n,K,A):
    coupling = np.zeros(n)
    for j in range(n):
        for i in range(n):
            coupling[j] += A[i,j]*np.sin(theta[i]-theta[j])

    return omega + (K/n)*coupling

#%%

T=100
t=np.linspace(0,T,2001);

n=len(L); # number of oscillators
#n=10

rad=np.ones((n,1));
thetai=2*np.pi*np.random.rand(n);
omega=np.random.rand(n)+0.5;

#A=np.random.rand(n,n);
#A[np.where(A<=0.5)] = 0;
#%%
Ks = [0,1,5,10,15,20,30,40,50,60,70,80,90,100]
ave_L_r = np.zeros(len(Ks))
ave_Ac_r = np.zeros(len(Ks))
ave_C_r = np.zeros(len(Ks))

for k,K in enumerate(Ks):
    print(k)
    
    # L
    sol = sp.integrate.solve_ivp(lambda t,theta:kura_rhs(t,theta,omega,n,K,L_adj),(0,T),thetai,t_eval=t)
    t = sol.t
    theta = sol.y
    
    r = np.abs(np.mean(np.exp(1j*theta),axis=0))
    ave_L_r[k] = np.mean(r[int(279/3):])
    
    plt.figure()
    plt.plot(t,theta.T%(2*np.pi),color='k',alpha=.01)
    plt.savefig('img/phase/L_'+str(K)+'.pdf')

    # Ac
    sol = sp.integrate.solve_ivp(lambda t,theta:kura_rhs(t,theta,omega,n,K,Ac_adj),(0,T),thetai,t_eval=t)
    t = sol.t
    theta = sol.y
    
    r = np.abs(np.mean(np.exp(1j*theta),axis=0))
    ave_Ac_r[k] = np.mean(r[int(279/3):])
    
    plt.figure()
    plt.plot(t,theta.T%(2*np.pi),color='k',alpha=.01)
    plt.savefig('img/phase/Ac_'+str(K)+'.pdf')

    
    # C
    sol = sp.integrate.solve_ivp(lambda t,theta:kura_rhs(t,theta,omega,n,K,C_adj),(0,T),thetai,t_eval=t)
    t = sol.t
    theta = sol.y
    
    r = np.abs(np.mean(np.exp(1j*theta),axis=0))
    ave_C_r[k] = np.mean(r[int(279/3):])
    
    plt.figure()
    plt.plot(t[::5],theta.T[::5]%(2*np.pi),color='k',alpha=.01)
    plt.savefig('img/phase/C_'+str(K)+'.pdf')
        
    
#%%

plt.figure()
plt.plot(Ks,ave_L_r,color='k',linestyle=':',marker='o')
plt.plot(Ks,ave_Ac_r,color='k',linestyle=':',marker='s')
plt.plot(Ks,ave_C_r,color='k',linestyle=':',marker='^')
plt.ylim([0,1])
plt.savefig('img/couple_strength.pdf')


#%%

plt.figure()
plt.plot(t,theta.T%(2*np.pi),color='k',alpha=.01)
#plt.ylabel('phase coherence')
#plt.xlabel('coupleing strength')
plt.savefig('img/phase_Ac_'+str(K)+'.pdf')

#%%
Ks = [1,10,20,30,40,50,75,100,150,200]
ave_Lg_r = np.zeros(len(Ks))
ave_Acg_r = np.zeros(len(Ks))

for k,K in enumerate(Ks):
    print(k)
    
    # L
    sol = sp.integrate.solve_ivp(lambda t,theta:kura_rhs(t,theta,omega,n,K,L_g),(0,T),thetai,t_eval=t)
    t = sol.t
    theta = sol.y
    
    r = np.abs(np.mean(np.exp(1j*theta),axis=0))
    ave_Lg_r[k] = np.mean(r[int(279/3):])
    
    plt.figure()
    plt.plot(t,theta.T%(2*np.pi),color='k',alpha=.01)
    plt.savefig('img/phase/Lg_'+str(K)+'.pdf')

    # Ac
    sol = sp.integrate.solve_ivp(lambda t,theta:kura_rhs(t,theta,omega,n,K,Ac_g),(0,T),thetai,t_eval=t)
    t = sol.t
    theta = sol.y
    
    r = np.abs(np.mean(np.exp(1j*theta),axis=0))
    ave_Acg_r[k] = np.mean(r[int(279/3):])
    
    plt.figure()
    plt.plot(t,theta.T%(2*np.pi),color='k',alpha=.01)
    plt.savefig('img/phase/Acg_'+str(K)+'.pdf')


#%%

plt.figure()
plt.plot(Ks,ave_Lg_r,color='k',linestyle=':',marker='o')
plt.plot(Ks[:-1],ave_Acg_r[:-1],color='k',linestyle=':',marker='s')
plt.ylim([0,1])
#plt.ylabel('phase coherence')
#plt.xlabel('coupleing strength')
plt.savefig('img/couple_strength_g.pdf')

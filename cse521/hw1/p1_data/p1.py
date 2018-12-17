#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:17:09 2018

@author: tyler
"""

import numpy as np
import sys

#%%
def karger(G,vertex_label,vertex_degree,size_V):
    
    size_V = len(vertex_label)
    #N = int(size_V*(1-1/np.sqrt(2)))
    iteration_schedule = [size_V-2]
    
    for N in iteration_schedule:
        for n in range(N):
#            if n%1000==0: print('iteration:',n)
            
            # uniformly at random pick e = (v0,v1)
            cs0 = np.cumsum(vertex_degree)
            rand_idx0 = np.random.randint(cs0[-1])
            e0 = np.searchsorted(cs0,rand_idx0,side='right')
            
            #cs1 = np.cumsum(np.append(G[e0,e0:],G[:e0,e0]))
            cs1 = np.cumsum(G[e0])
            rand_idx1 = np.random.randint(vertex_degree[e0])
            e1 = np.searchsorted(cs1,rand_idx1,side='right')
            
            if(G[e0,e1] == 0):
                print('picked empty edge')
            
            v0 = e0
            v1 = e1
            
            # bring edges from v1 into v0
            
            # add new edges to v0
            G[v0] += G[v1]
            G[:,v0] += G[v1]
            new_edge_count = vertex_degree[v1] - G[v0,v0] #- G[v1,v1]
            
            # delete old edges from v1
            G[v1] = 0
            G[:,v1] = 0
            
            # delete any created loops 
            G[v0,v0] = 0
            
            np.putmask(vertex_label,vertex_label==v1,v0)
       
            
            vertex_degree[v0] += new_edge_count 
                
            vertex_degree[v1] = 0
    
    nz = np.nonzero(vertex_degree)[0]
    
    if(len(nz) != 2):
        print('did not find well defined cut')
    
    SN0 = np.where(vertex_label == nz[0])[0]
    SN1 = np.where(vertex_label == nz[1])[0]
    
    if len(SN0) + len(SN1) != size_V:
        print('lost nodes')
    
    if len(SN0) < len(SN1):
        cut = SN0
    else:
        cut = SN1
    
    return cut,vertex_degree[nz[0]]
        
#%%

#python p1.py z N ID
z = sys.argv[1] # 0,1,2,3 
N = int(sys.argv[2]) # integer number of runs
ID = sys.argv[3] # output file id

#%%
E_raw = np.loadtxt('b'+str(z)+'.in',dtype='int')

min_E = np.min(E_raw)
E = E_raw - min_E
size_V = np.max(E)+1

G = np.zeros((size_V,size_V),dtype='int64')
vertex_degree = np.zeros(size_V,dtype='int64')
for e0,e1 in E:
    vertex_degree[e0] += 1;
    vertex_degree[e1] += 1;
    G[min(e0,e1),max(e0,e1)] += 1;
    G[max(e0,e1),min(e0,e1)] += 1;


vertex_label = np.arange(size_V,dtype='int64') # gives index of supervertex containg vertex


#%%
f=open('b'+z+'/cuts_'+ID+'.dat','ab')
g=open('b'+z+'/cut_sizes_'+ID+'.dat','ab')
#
for n in range(N):
    if n%500 == 0:
        print(ID+'_trial :', n,' of ',N)
    vl,cut_size = karger(np.copy(G),np.copy(vertex_label),np.copy(vertex_degree),size_V)    
    np.savetxt(f,[vl],fmt='%d',delimiter=',')
    np.savetxt(g,[cut_size],fmt='%d',delimiter=',')

f.close()
g.close()

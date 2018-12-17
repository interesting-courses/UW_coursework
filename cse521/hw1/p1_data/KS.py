#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:17:09 2018

@author: tyler
"""

import numpy as np
import sys

#%%

def karger(G,vertex_label,vertex_degree,iteration_schedule):
    
    if len(iteration_schedule) > 14:
        print('='*len(iteration_schedule),len(iteration_schedule))
        
    for n in range(iteration_schedule[0]):
#            if n%1000==0: print('iteration:',n)
        
        # uniformly at random pick e = (v0,v1)
        cs0 = np.cumsum(vertex_degree)
        rand_idx0 = np.random.randint(cs0[-1])
        v0 = np.searchsorted(cs0,rand_idx0,side='right')
        
        #cs1 = np.cumsum(np.append(G[e0,e0:],G[:e0,e0]))
        cs1 = np.cumsum(G[v0])
        rand_idx1 = np.random.randint(cs1[-1])
        v1 = np.searchsorted(cs1,rand_idx1,side='right')
        

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

        # update degrees
        vertex_degree[v0] += new_edge_count 
        vertex_degree[v1] = 0

        # update supernodes        
        np.putmask(vertex_label,vertex_label==v1,v0)
       
    nz = np.nonzero(vertex_degree)[0]
        
    if(len(nz) == 2):
        SN = [set(np.where(vertex_label == nz[0])[0]),set(np.where(vertex_label == nz[1])[0])]
        if vertex_degree[nz[0]] < 100: # assume min cut is of size  <100
            return [SN],vertex_degree[nz[0]]
        else:
            return [],100
    
    else:
        cuts_H = []
        size_V_H = len(nz)
        H = G[np.ix_(nz,nz)]
        vertex_label_H = np.arange(size_V_H,dtype='int')
        
        SN0,v_d_0 = karger(np.copy(H),np.copy(vertex_label_H),vertex_degree[nz],iteration_schedule[1:])
        SN1,v_d_1 = karger(np.copy(H),np.copy(vertex_label_H),vertex_degree[nz],iteration_schedule[1:])
              
        # keep all min cuts
        cut_size = min(v_d_0,v_d_1)
        if v_d_0 == cut_size:
            cuts_H += SN0
        if v_d_1 == cut_size:
            cuts_H += SN1
        
        cuts = []
        for cut in cuts_H:
            # build updated list of supernodes
            SN = [set([]),set([])]
            for i,n in enumerate(nz):
                eq_nodes = np.where(vertex_label == n)[0]
                if i in cut[0]:
                    for j in eq_nodes:
                        SN[0].add(j)
                else:
                    for j in eq_nodes:
                        SN[1].add(j)

            cuts.append(SN)
        
    return cuts,cut_size
        
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

G = np.zeros((size_V,size_V),dtype='uint16')
vertex_degree = np.zeros(size_V,dtype='int')
for e0,e1 in E:
    vertex_degree[e0] += 1;
    vertex_degree[e1] += 1;
    G[min(e0,e1),max(e0,e1)] += 1;
    G[max(e0,e1),min(e0,e1)] += 1;

del(E)
del(E_raw)

vertex_label = np.arange(size_V,dtype='int') # gives index of supervertex containg vertex
#%%
c = np.sqrt(2)
iter_schedule = [int(size_V/c**i) for i in range(int(np.floor(np.log(size_V/6)/np.log(c))))]
iter_schedule.append(2)
iter_schedule.reverse()
iter_schedule = np.diff(iter_schedule)[::-1]
iter_schedule[-1] += size_V-2 - np.sum(iter_schedule)

#%%
#cuts,cut_size = karger(np.copy(G),np.copy(vertex_label),np.copy(vertex_degree),iter_schedule)

#%%
f=open('b'+z+'/cuts_'+ID+'.dat','ab')
g=open('b'+z+'/cut_sizes_'+ID+'.dat','ab')
#
for n in range(N):
    if n%1 == 0:
        print(ID+'_trial :', n+1,' of ',N)
    cuts,cut_size = karger(np.copy(G),np.copy(vertex_label),np.copy(vertex_degree),iter_schedule)
    for cut in cuts:
        if len(cut[0]) < len(cut[1]):
            vl = list(cut[0])
        else:
            vl = list(cut[1])
        np.savetxt(f,[np.sort(vl)],fmt='%d',delimiter=',')
        np.savetxt(g,[cut_size],fmt='%d',delimiter=',')

f.close()
g.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:57:02 2018

@author: tyler
"""

import numpy as np

#%%
z=3
E_raw = np.loadtxt('b'+str(z)+'.in',dtype='int')


#%%
min_E = np.min(E_raw)
E = E_raw - min_E
size_V = np.max(E)+1
del(E_raw)

V = np.zeros((size_V,int(len(E)/2)),dtype='uint16') # gives all vertices attached to a given vertex
vertex_degree = np.zeros(size_V,dtype='uint16') # gives degree of supervertices
vertex_label = np.arange(size_V,dtype='uint16') # gives index of supervertex containg vertex

for e0,e1 in E:
    V[e0,vertex_degree[e0]] = e1
    vertex_degree[e0] += 1
    
    V[e1,vertex_degree[e1]] = e0
    vertex_degree[e1] += 1

del(E)

SN = np.array([set([v]) for v in range(size_V)]) #supernodes

#%%
def karger(V,SN,vertex_degree,vertex_label):

    size_V = len(SN)
    #N = int(size_V*(1-1/np.sqrt(2)))
    iteration_schedule = [5000]
    
    for N in iteration_schedule:
        for n in range(N):
            #================================
            #  get random edge e = v0,v1    
            #================================
            if n%500==0: 
                print('---------------')
                print('iteration: ',n)
    
            # pick e0 among supervertices weighted by degree
            v0 = np.random.choice(size_V,p=vertex_degree/np.sum(vertex_degree))
            
            # pick random edge from e0 and find correct supervertex
            v1 = vertex_label[np.random.choice(V[v0,:vertex_degree[v0]])]
            
            
            # bring edges of e1 into e0
            v0_verts = V[v0,:vertex_degree[v0]]
            v1_verts = V[v1,:vertex_degree[v1]]
            
            sn0 = SN[v0]
            sn1 = SN[v1]
            
            vertex_label[list(sn1)] = v0
        
            
            clean_v0_verts = [v for v in v0_verts if v not in sn1]
            clean_v1_verts = [v for v in v1_verts if v not in sn0]
            
            # number of cleaned vertices from each side should be equal
    #        if(len(e0_verts)-len(clean_e0_verts) != len(e1_verts)-len(clean_e1_verts)):
    #            print('error in iteration ',n)
    #            return SN,vertex_degree,vertex_label
            
            # add vertices back to e0
            V[v0,:len(clean_v0_verts)] = clean_v0_verts
            V[v0,len(clean_v0_verts):len(clean_v0_verts)+len(clean_v1_verts)] = clean_v1_verts
            
            # update vertex degrees
            vertex_degree[v0] = len(clean_v0_verts)+len(clean_v1_verts)
            vertex_degree[v1] = 0
        
            SN[v0] = sn0 | sn1
    
        # clean supervertices
        for v0 in range(size_V):
            for v1 in range(vertex_degree[v0]):
                V[v0,v1] = vertex_label[V[v0,v1]]
        SN = np.array([set([v]) for v in range(size_V)])
        
        # IF WE BOOST NOW IS A GOOD TIME

    return V,SN,vertex_degree,vertex_label

#%%

V_ = np.copy(V)
SN_ = np.copy(SN)
vertex_degree_ = np.copy(vertex_degree)
vertex_label_ = np.copy(vertex_label)

#%%

V_,SN_,vertex_degree_,vertex_label_ = karger(V_,SN_,vertex_degree_,vertex_label_)

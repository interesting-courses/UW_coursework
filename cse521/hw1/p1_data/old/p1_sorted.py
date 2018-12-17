#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 23:54:49 2018

@author: tyler
"""

import numpy as np
import cProfile
import time

#%%
def preprocess_E(E_in):
    '''
    returns : renaming of edges in E so that every edge is unqiue. 
              vertex equivalence classes
              permutation to sort edges by second vertex
              permutation to undo above sort
    '''
    E = np.copy(E_in)
    E = E[np.argsort(100*E[:,0]+E[:,1])]
    
    V = np.unique(E)
    supernodes = np.array([set([v]) for v in V])
    supernode_nonempty_Q = np.ones(len(V),dtype='bool')

    v = np.max(V)+1
    
    i = 0
    e = [-1,-1]
    while i < len(E):
        e_ = e
        e = E[i]
        if np.all(e == e_):            
            # add new vertex to equivalence classes
            for j in range(len(V)):
                if e[1] in supernodes[j]:
                    supernodes[j] = supernodes[j] | {v}

            e = E[i-1]
            E[i,1] =  v
            v += 1
        i += 1

    right_sort = np.argsort(E[:,1])
    
    print("number of duplicate edges:", v - np.max(V) - 1)

    return E,E[right_sort],supernodes,supernode_nonempty_Q,right_sort


#%%
def karger(E,E_rs,supernodes,supernode_nonempty_Q,right_sort):
    start = time.clock()

    E_l = E[:,0]
    E_r = E_rs[:,1]
    
    size_E = np.shape(E)[0]
    size_V = sum(supernode_nonempty_Q)
    
    # True if edge is not a loop
    not_loop_Q = np.ones(size_E,dtype='bool')
    
    f = 0
    s = 0
    for j in range(size_V-2):
        if j%1==0:
            print('iteration:', j)
            print('find endpoints: ',f)
            print('find loops    : ', s)
            f = 0
            s = 0
            sn_count = sum(map(len,supernodes[np.where(supernode_nonempty_Q)[0]]))
            print('numer of vert :', sn_count)
            print('numer of sns  :', size_E)
        
        # pick random edge
        cs = np.cumsum(not_loop_Q)
        size_E = np.sum(not_loop_Q)
        rand_idx = np.where(cs > np.random.randint(size_E))[0][0]
        e0,e1 = E[rand_idx]
        print('e:', e0,',',e1)        
        
        #find edge endopoint vertices

        start = time.clock()
        supernode_nonempty_idx = np.where(supernode_nonempty_Q)[0]
        
        for i0 in supernode_nonempty_idx: 
            if e0 in supernodes[i0]:
                break
            
        for i1 in supernode_nonempty_idx:
            if e1 in supernodes[i1]:
                break
        f += (time.clock() - start)

        if i0==i1:

            print('sn0: ',supernodes[i0])
            print('sn1: ',supernodes[i1])
            return        

        # merge vertex equivalence classes
        sn0 = supernodes[i0]
        sn1 = supernodes[i1]
        
        # find loops
        # search for edges with one end in sn0 and one in sn1
        start = time.clock()

        for n in sn0:
            left_low_idx = np.searchsorted(E_l,n,side='left')
            left_high_idx = np.searchsorted(E_l,n,side='right')
            for i in range(left_low_idx,left_high_idx):
                if (E[i,1] in sn1) and not_loop_Q[i]:
                    size_E -= 1
                    not_loop_Q[i] = False
            
            right_low_idx = np.searchsorted(E_r,n,side='left')
            right_high_idx = np.searchsorted(E_r,n,side='right')
            for i in range(right_low_idx,right_high_idx):
                rsi = right_sort[i]
                if (E_rs[i,0] in sn1) and not_loop_Q[rsi]:
                    size_E -= 1
                    not_loop_Q[rsi] = False
        
        s += time.clock() - start
        
        # put sn1 into sn0 and sn1 into sn0 and delete sn1

        
        print(supernodes[i0])
        print(supernodes[i1])
        supernodes[i0] = supernodes[i0] | supernodes[i1]
        np.delete(supernodes,i1)
        np.delete(supernode_nonempty_Q,i1)
#        supernode_nonempty_Q[i1] = False
        
        print(supernodes[np.where(supernode_nonempty_Q)[0]])
        
        
    return supernodes[np.where(supernode_nonempty_Q)[0]],sum(not_loop_Q)
                  
#%%
E_raw = np.loadtxt("b1.in",dtype='int')

#%%
E_,E_rs_,supernodes_,supernode_nonempty_Q_,right_sort_ = preprocess_E(E_raw)    

#%%

E,E_rs,supernodes,supernode_nonempty_Q,right_sort = np.copy(E_),np.copy(E_rs_),np.copy(supernodes_),np.copy(supernode_nonempty_Q_),np.copy(right_sort_)

karger(E,E_rs,supernodes,supernode_nonempty_Q,right_sort)
#cProfile.run('karger(E,np.copy(supernodes),np.copy(supernode_nonempty_Q),right_sort)')
#%%

E[np.any(E==2,axis=1)]

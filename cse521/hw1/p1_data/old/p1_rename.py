#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 23:54:49 2018

@author: tyler
"""

import numpy as np

#%%
    
def postprocess_cut(supernodes_original,supernodes_f,supernode_nonempty_Q,not_loop_Q):
    '''
    returns : partition of original vertices of $G$ and size of coresponding cut
    '''
    sn = supernodes_f[supernode_nonempty_Q]
    size_E = sum(not_loop_Q)
    
    if len(sn[0])<len(sn[1]):
        sn = sn[0]
    else:
        sn = sn[1]
    
    for v in list(sn):
        if v >= len(supernodes):
            sn.remove(v)
            for sn_o in supernodes_original:
                if v in sn_o:
                    sn.add(min(sn_o))
                    break
    
    return list(sn),size_E


#%%
def karger(E,supernodes,supernode_counts,not_loop_Q):
#    start = time.clock()
    
    size_EE = np.shape(E)[0]
    size_E = np.shape(E)[0]
    size_V = len(supernodes)
    

    for j in range(size_V-2):
        if j%10==0:
            print('===================')
            print('iteration:', j)

            
        
        # pick random edge
        cs = np.cumsum(not_loop_Q) 
#        rand_idx = np.where(cs > np.random.randint(cs[-1]))[0][0]
        rand_idx = np.searchsorted(cs, np.random.randint(cs[-1]))
        e0,e1 = E[rand_idx]
        
        # update partition
        
        size_v0 = supernode_counts[e0]
        size_v1 = supernode_counts[e1]
        
        if size_v1 == 0 or e0==e1:
            print("e0,e1: ",e0,e1)
            print(rand_idx)
            return E,supernodes,supernode_counts,not_loop_Q


        supernodes[e0,size_v0:size_v0+size_v1] = supernodes[e1,:size_v1]
        supernode_counts[e0] += size_v1
        supernode_counts[e1] = 0
        

        
        # relabel edges and delete
        for k in range(size_EE):            
            if not_loop_Q[k]:
                if E[k,0] == e1:
                    E[k,0] = e0
                    if E[k,1] == e0:
                        not_loop_Q[k] = False
                    elif E[k,1] == e1:
                        E[k,1] = e0
                        not_loop_Q[k] = False
                if E[k,1] == e1:
                    E[k,1] = e0
                    if E[k,0] == e0:
                        not_loop_Q[k] = False

        
        #probably can't be faster than this unless we can compile
        

    return E,supernodes,supernode_counts,not_loop_Q



#%%

E = np.loadtxt('b3.in',dtype='int')-1

#E = np.array([[1,2],[1,2],[1,2],[1,3]])

not_loop_Q = np.ones(len(E),dtype='bool')


size_V = np.max(np.unique(E))+1

supernodes = np.zeros((size_V,size_V),dtype='int')
supernodes[:,0] = np.arange(size_V,dtype='int')
supernode_counts = np.ones(size_V,dtype='int')

#%%
E_out,supernodes,supernode_counts,not_loop_Q = karger(np.copy(E),supernodes,supernode_counts,not_loop_Q)

#%%

postprocess_cut(supernodes_,supernodes,supernode_nonempty_Q,not_loop_Q)

#%%

len(E[np.any(E==8,axis=1)])

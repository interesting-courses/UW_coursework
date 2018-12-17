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
def karger(E,G,supernodes,supernode_nonempty_Q,not_loop_Q):
#    start = time.clock()
    
    size_E = np.shape(E)[0]
    size_V = sum(supernode_nonempty_Q)  
    
    f = 0
    s = 0
    
    sn0 = {}
    for j in range(size_V-2):
        if j%500==0:
            print('===================')
            print('iteration:', j)
            '''
            print('find endpoints: ',f)
            print('find loops    : ', s)
            f = 0
            s = 0
            print('sn0 size: ', len(sn0))
            sn_count = sum(map(len,supernodes[np.where(supernode_nonempty_Q)[0]]))
            print('numer of vert :', sn_count)
            print('numer of sns  :', sum(supernode_nonempty_Q))
            '''
            
        
        # pick random edge

        #probably can't be faster than this unless we can compile
        cs = np.cumsum(not_loop_Q) 
#        rand_idx = np.where(cs > np.random.randint(cs[-1]))[0][0]
        rand_idx = np.searchsorted(cs, np.random.randint(cs[-1]))
        e0,e1 = E[rand_idx]
            
        #find edge endopoint vertices

#        start = time.clock()
        
        
        supernode_nonempty_idx = np.where(supernode_nonempty_Q)[0]
        for i0 in supernode_nonempty_idx: 
            if e0 in supernodes[i0]:
                break
            
        for i1 in supernode_nonempty_idx[::-1]:
            if e1 in supernodes[i1]:
                break
#        f += (time.clock() - start)
        
        # merge vertex equivalence classes
        sn0 = supernodes[i0]
        sn1 = supernodes[i1]
        
        # find loops
        # search for edges with one end in sn0 and one in sn1
#        start = time.clock()
        
        for i in sn0:
            Gi = G[i]
            for j in sn1:
                Gij = Gi[j]
                if Gij != -1:
                    if not_loop_Q[Gij]:
                        not_loop_Q[Gij] = False
                    
#        s += time.clock() - start
        
        # put sn1 into sn0 and sn1 into sn0 and delete sn1
        supernodes[i0] = supernodes[i0] | supernodes[i1]
        supernode_nonempty_Q[i1] = False

    return supernodes,supernode_nonempty_Q,not_loop_Q


#%% load data

d = np.load('b0_pre.npz')

E=d['E']
G=d['G']
supernodes_=d['supernodes_']
supernode_nonempty_Q_=d['supernode_nonempty_Q_']

del(d)


#%%

supernodes,supernode_nonempty_Q = np.copy(supernodes_),np.copy(supernode_nonempty_Q_)
not_loop_Q = np.ones(len(E),dtype='bool')

supernodes,supernode_nonempty_Q,not_loop_Q = karger(E,G,supernodes,supernode_nonempty_Q,not_loop_Q)

#%%

postprocess_cut(supernodes_,supernodes,supernode_nonempty_Q,not_loop_Q)

#%%

len(E[np.any(E==8,axis=1)])

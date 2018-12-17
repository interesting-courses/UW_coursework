#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 23:54:49 2018

@author: tyler
"""

import numpy as np

#%%
    
def postprocess_cut(supernodes_original,supernodes_f,supernode_nonempty_Q,edge_counts):
    '''
    returns : partition of original vertices of $G$ and size of coresponding cut
    '''
    sn = supernodes_f[supernode_nonempty_Q]
    size_E = sum(edge_counts)
    
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
    
    return sn,size_E

#%%

def clean_graph():
    return -1

#%%
@profile
def karger(E,G,edge_counts,supernodes,supernode_nonempty_Q,N):
        
    vertex_label = np.arange(len(supernodes)) #i-th entry gives index of supernode containing vertex i

    for n in range(N):
        if n%500==0:
            print('----------------')
            print('iteration:', n)
            '''
            v_count = sum(map(len,supernodes[np.where(supernode_nonempty_Q)[0]]))
            print('numer of vert :', v_count)
            print('numer of sns  :', sum(supernode_nonempty_Q))
            '''
        
        
        ##### pick random edge #####
        
        #probably can't be faster than this unless we can compile
        cs = np.cumsum(edge_counts) 
        rand_idx = np.searchsorted(cs, np.random.randint(cs[-1]))
        e0,e1 = E[rand_idx]
                    
        #find edge endopoint vertices
        i0 = vertex_label[e0]
        i1 = vertex_label[e1]
        
        
        ###### merge vertex equivalence classes #####
        
        sn0 = supernodes[i0]
        sn1 = supernodes[i1]
        
        # update supervertex containment

        vertex_label[list(sn1)] = i0

        ##### remove loops #####

        # search for edges with one end in sn0 and one in sn1
        for i in sn0:
            Gi = G[i]
            for j in sn1:
                Gij = Gi[j]
                if (Gij != -1) and edge_counts[Gij]:
                    edge_counts[Gij] = 0
                    
        # update supervertices
        supernodes[i0] = supernodes[i0] | supernodes[i1]
        supernode_nonempty_Q[i1] = False

    return supernodes,supernode_nonempty_Q,edge_counts,vertex_label


#%% load data
z=3

d = np.load('b'+str(z)+'_pre1.npz')
E=d['E']
G=d['G']
edge_counts_ = d['edge_counts']
supernodes_=d['supernodes']
supernode_nonempty_Q_ = d['supernode_nonempty_Q']

del(d)


#%%

edge_counts,supernodes,supernode_nonempty_Q = np.copy(edge_counts_),np.copy(supernodes_),np.copy(supernode_nonempty_Q_)

size_V = len(G)
N = size_V - 2

sn,sn_n_Q,e_c,v_l= karger(E,G,np.copy(edge_counts_),np.copy(supernodes),np.copy(supernode_nonempty_Q),N)
'''
#%%

K=300

f=open('b'+str(z)+'_cuts.dat','ab')
g=open('b'+str(z)+'_cut_sizes.dat','ab')
for k in range(K):
    print('============================')
    print('boost:', k)
    sn,sn_n_Q,e_c,v_l= karger(E,G,np.copy(edge_counts_),np.copy(supernodes),np.copy(supernode_nonempty_Q),N)
    cut,size = postprocess_cut(supernodes_,sn,sn_n_Q,e_c)
    
    np.savetxt(f,[list(cut)],fmt='%d',delimiter=',')
    np.savetxt(g,[size],fmt='%d',delimiter=',')

f.close()
g.close()


#%%
#be careful to use not preprocessed E for this.
len(E[np.any(E==22,axis=1)])
'''
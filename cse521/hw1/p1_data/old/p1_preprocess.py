import numpy as np


#%%
def preprocess_E(E_in):
    '''
    returns : renaming of edges in E so that every edge is unqiue. 
              vertex equivalence classes
              permutation to sort edges by second vertex
              permutation to undo above sort
    '''
    E = np.copy(E_in)
    E = np.sort(E,axis=1) # edges always have lower vertex first
    E = E[np.argsort(10000*E[:,0]+E[:,1])] # now sort edges to help find duplicates
    
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
    
    print('construct G')
    G = -1*np.ones((v+1,v+1),dtype='int')
    for k in range(np.shape(E)[0]):
        e0,e1 = E[k]
        G[e0,e1] = k
        G[e1,e0] = k
            
    print("number of duplicate edges:", v - np.max(V) - 1)

    return E,G,supernodes,supernode_nonempty_Q

#%%
    
for i in [0,1,2,3]:
    E_raw = np.loadtxt('b'+str(i)+'.in',dtype='int')
    E = preprocess_E(E_raw)
    np.savez('b'+str(i)+'_pre',E=E,G=G,supernodes=supernodes,supernode_nonempty_Q=supernode_nonempty_Q)

#%%
def preprocess1_E(E_in):
    '''
    returns : renaming of edges in E so that every edge is unqiue. 
              vertex equivalence classes
              permutation to sort edges by second vertex
              permutation to undo above sort
    '''
    print('construct E')
    E_ = np.copy(E_in)
    E_ = np.sort(E_,axis=1) # edges always have lower vertex first
    E_int = 10000*E_[:,0]+E_[:,1] # now sort edges to help find duplicates
    
    E_int_unique,index,edge_counts = np.unique(E_int,return_index=True,return_counts=True)    
    
    E = E_[index]
    
    print('construct V')
    V = np.unique(E)
    minV = np.min(V)
    
    V -= minV
    E -= minV
    
    size_V = len(V)
    supernodes = np.array([set([v]) for v in V])
    supernode_nonempty_Q = np.ones(size_V,dtype='bool')
        
    print('construct G')
    G = -1*np.ones((np.max(V)+1,np.max(V)+1),dtype='int')
    for k in range(len(E)):
        e0,e1 = E[k]
        G[e0,e1] = k
        G[e1,e0] = k
        
    return E,G,edge_counts,supernodes,supernode_nonempty_Q

#%% Preprocess Data

for i in [0,1,2,3]:
    E_raw = np.loadtxt('b'+str(i)+'.in',dtype='int')
    E,G,edge_counts,supernodes,supernode_nonempty_Q = preprocess1_E(E_raw)
 #   np.savez('b'+str(i)+'_pre1',E=E,G=G,edge_counts=edge_counts,supernodes=supernodes,supernode_nonempty_Q=supernode_nonempty_Q)
    np.savetxt('b'+str(i)+'_min_cut_size.dat',[len(G)],fmt='%d')

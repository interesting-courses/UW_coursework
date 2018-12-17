#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt

#%%

#<start:import>
# load file
z=0
f = np.fromfile('j'+str(z)+'.in',sep=' ',dtype='int')

# format data
n,size_U = f[:2]
A = np.reshape(f[2:],(n,-1))-1
del(f)

unique = np.unique(A)
size_unique = len(unique)
#<end:import>

#%%

L = 5000 # number of hash functions

h_A = np.zeros((n,L),dtype='float32')

h = np.zeros(size_U,dtype='float32')
#h[unique] = np.random.randint(2**32,size=(size_unique))
#h[unique] = np.random.rand(size_unique)

# writing this many things to memory is slow....

print('builuding hash functions')
#<start:make_hash>
for l in range(L):
    if l%50==0:
        print(l)

    # generate hash function
    h[unique] = np.random.rand(size_unique)
    h_A[:,l] = np.min(h[A],axis=1)
    
#<end:make_hash>

#%%    
f=open('h_A'+str(z)+'.dat','ab')
np.savetxt(f,h_A.T,fmt='%1.16f',delimiter=',')
f.close()

h_A = np.loadtxt('h_A'+str(z)+'.dat',delimiter=',').T

#%%

print('comparing hash values')
#<start:comp_hash>

jac_hash=[]
for i in range(n):
    for j in range(i+1,n):
        collisions = (h_A[i]==h_A[j])
        jac_hash.append(np.mean(collisions))
#<end:comp_hash>
        
np.savetxt('j'+str(z)+'.out',np.transpose([jac_hash]),fmt='%1.2e',delimiter=',')


#%% 

# exact Jaccard
#<start:exact>
A_set = []
for i in range(n):
    A_set.append(frozenset(A[i]))

jac_exact = []
for i in range(n):
    if i%10==0:
        print(i)
    for j in range(i+1,n):
        size_intersection = len(A_set[i]&A_set[j])
        size_union = len(A_set[i]|A_set[j])
        jac_exact.append( size_intersection/size_union )
#<end:exact>

np.savetxt('j'+str(z)+'.check',np.transpose([jac_exact]),fmt='%1.2e',delimiter=',')

#%%

jac_hash = np.loadtxt('j'+str(z)+'.out')
jac_exact = np.loadtxt('j'+str(z)+'.check')

#%%
x_min = .9*np.min(jac_exact)
x_max = 1.1*np.max(jac_exact)
L = np.shape(h_A)[1]

error = 1-np.divide(jac_hash, jac_exact, out=np.zeros_like(jac_hash), where=jac_exact!=0)

plt.figure(figsize=(10,5))
#plt.fill_between([0,n*(n-1)/2],[1.1,1.1],[.9,.9], facecolor='red', alpha=0.25)
#plt.plot(np.arange(n*(n-1)/2),error,marker='.',linestyle='None',color='0')
plt.axvspan(-.1,.1, alpha=0.25, color='red')

plt.hist(error,15,color='0')

plt.title('L = $'+str(L)+'$')
plt.ylabel('count')
plt.xlabel('relative error')

plt.savefig('img/error_hist'+str(z)+'.pdf')


#%%

xy = np.unique(np.vstack([jac_exact,jac_hash]),axis=1)

plt.figure(figsize=(10,5))

plt.fill_between([0,x_max],[0,1.1*x_max],[0,.9*x_max], facecolor='red', alpha=0.25)
#plt.fill_between([0,x_max],[.1,.1+x_max],[-.1,-.1+x_max], facecolor='red', alpha=0.25)

plt.scatter(xy[0],xy[1],marker='.',color='0')

plt.title('L = $'+str(L)+'$')
plt.ylabel('approximate Jaccard distance')
plt.xlabel('exact Jaccard distance')

plt.savefig('img/error'+str(z)+'.pdf')


#%%


def jac(A1,A2,L):
    k=0
    h = np.zeros(size_U)
    for t in range(L):
        h[A1] = np.random.rand(len(A1))
        h[A2] = np.random.rand(len(A2))
   
        if np.min(h[A1]) == np.min(h[A2]):
            k+=1
    return k/L

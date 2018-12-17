#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import imageio
import matplotlib.pyplot as plt
import timeit


#%%
#<start:svd>
def compress_image(A,k):
    u,s,vt = np.linalg.svd(A, full_matrices=False)
    
    return (u[:,:k]*s[:k])@vt[:k]
#<end:svd>
#%%
#<start:approx_svd>
def col_sample_compress_image(A,k,eps):
    m,n = np.shape(A)
    
    column_norms = np.linalg.norm(A,axis=0)
    
    column_probs = column_norms**2
    column_probs /= np.sum(column_probs)
    
    s = int(k/eps**2)
    
    indices = np.random.choice(np.arange(n), size=s, p=column_probs)
    
    X = A[:,indices]
    
    u,s,vt = np.linalg.svd(X, full_matrices=False)
    
    return u[:,:k]@(u[:,:k].T@A)
#<end:approx_svd>

#%%
#<start:approx_qr>
def gaus_sample_compress_image(A,k):
    m,n = np.shape(A)
    
    X = A@np.random.randn(n,k)
    u,s,vt = np.linalg.svd(X, full_matrices=False)
    
    return u[:,:k]@(u[:,:k].T@A)
#<end:approx_qr>

#%%
A = imageio.imread('einstein.jpg')[:,:,0]
K = np.logspace(0,2.7,8,dtype='int')
for i,k in enumerate(K):
    A_k = compress_image(A,k)
    plt.imsave('img/'+str(k)+'.png', A_k[::6,::6], format='png')
    
    A_k_col = col_sample_compress_image(A,k,1)
    plt.imsave('img/'+str(k)+'_fast.png', A_k_col[::6,::6], format='png')
    
    A_k_gaus = gaus_sample_compress_image(A,k)
    plt.imsave('img/'+str(k)+'_faster.png', A_k_gaus[::6,::6], format='png')

#%%
    
N = 1

K = np.logspace(0,2.7,8,dtype='int')
setup = 'from __main__ import A, compress_image'
setup_col = 'from __main__ import A, col_sample_compress_image'
setup_gaus = 'from __main__ import A, gaus_sample_compress_image'

t = np.zeros(len(K))
t_col = np.zeros(len(K))
t_gaus = np.zeros(len(K))

for i,k in enumerate(K):
    t[i] = timeit.timeit('compress_image(A,'+str(k)+')', number=N, setup=setup)/N
    t_col[i] = timeit.timeit('col_sample_compress_image(A,'+str(k)+',1)', number=N, setup=setup_col)/N
    t_gaus[i] = timeit.timeit('gaus_sample_compress_image(A,'+str(k)+')', number=N, setup=setup_gaus)/N

#%%
plt.figure(figsize=(10,6))
plt.scatter(K,t,label='exact')
plt.scatter(K,t_col,label='column sampling')
plt.scatter(K,t_gaus,label='gaussian sampling')

plt.xlabel('$k$')
plt.ylabel('time (s)')
plt.legend()

plt.xscale('log')
plt.yscale('log')
plt.savefig('img/times.pdf')



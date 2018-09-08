#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.io import netcdf
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
from IPython.display import clear_output

# import netCDF files and set up as numpy arrays
sst_nc = netcdf.netcdf_file('sst.wkmean.1990-present.nc', 'r')
mask_nc = netcdf.netcdf_file('lsmask.nc', 'r')

time = sst_nc.variables['time'][:].copy()
sst = np.transpose(sst_nc.variables['sst'][:].copy(),axes=[1,2,0])
mask = mask_nc.variables['mask'][:].copy()[0,:,:]

sst_nc.close()
mask_nc.close()

N = 360*180;
L = 1400;

M,N,T = np.shape(sst)
max_temp = np.max(sst)
min_temp = np.min(sst)

# apply mask to all entries
sst = sst*np.transpose([mask]*T,axes=[1,2,0])


#%%
#<start:generate_wavelets>
#wavelet defined on [0,1]
def Haar_wavelet(x,N):
    return 1 if x<1/2 else -1
Haar_wavelet = np.vectorize(Haar_wavelet)

def sin_wavelet(x,N):
    return 2* np.sin(2*np.pi*x*(N-1)/N)
sin_wavelet = np.vectorize(sin_wavelet)


# generate discrete wavelet basis based on mother wave w over 2^k points
# |w|^2 should be normalized on [0,1]
def generate_wavelet_basis(w,k):
    N = 2**k

    # rows of U are basis
    U = np.zeros((N,N))
    U[0] = np.ones(N) / np.sqrt(N)

    for i in range(k):
        for j in range(2**i):
            index = 2**i+j
            length = 2**(k-i)
            U[index,j*length:(j+1)*length] = w(np.linspace(0,1,length),length) / np.sqrt(length)

    return U
#<end:generate_wavelets>
#%% Plot Haar Wavelet basis for R^8

U_cont = generate_wavelet_basis(Haar_wavelet,10)
U_disc = generate_wavelet_basis(Haar_wavelet,3)

for i in range(8):
    plt.figure()
    plt.plot(np.linspace(0,1,len(U_cont)),U_cont[i]/np.max(U_cont[i]),color='.8')
    plt.plot(np.linspace(0,8/9,len(U_disc)),U_disc[i]//np.max(U_disc[i]),color='k',marker='o',linestyle='None')
    plt.yticks=[-1,1]
    plt.ylim((-1.25,1.25))
    plt.savefig('img/haar/basis_'+str(i)+'.pdf')


#%%
#<start:wavelet_decomp>
k1 = 5
k2 = 5
k3 = 5

trimmed_sst = sst[0:2**k1,0:2**k2,0:2**k3]

U1 = generate_wavelet_basis(Haar_wavelet,k1)
U2 = generate_wavelet_basis(Haar_wavelet,k2)
U3 = generate_wavelet_basis(Haar_wavelet,k3)

N1 = 2**k1
N2 = 2**k2
N3 = 2**k3

coeffs = np.zeros(N1*N2*N3)

for i in range(N1):
    for j in range(N2):
        for k in range(N3):
            index = i*N2*N3 + j*N3 + k
            
            if index % 1000 == 0 : print(index)
            
            # construct 3d wavelet basis
            spatial_basis = np.transpose([U2[j]])@[U1[i]]
            full_basis = np.kron(np.transpose([[U3[k]]]),spatial_basis)

            #normalize
            full_basis /= np.linalg.norm(full_basis)
            
            full_basis = np.transpose(full_basis, axes=[2,1,0])
            
            coeffs[index] = np.sum(trimmed_sst*full_basis)
#<end:wavelet_decomp>
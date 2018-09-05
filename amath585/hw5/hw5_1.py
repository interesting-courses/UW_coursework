#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)

#<startTeX>
def five_point_poisson(m,img_q=False):
    # define RHS function
    def f(x,y): return x**2 + y**2

    # define boundary conditions
    bd = np.ones((m+2,m+2))
    bd[1:m+1,1:m+1] = np.zeros((m,m)) #must be zero on interior for below
    
    x = np.linspace(0,1,m+2)
    h = 1/(m+1)
    
    # construct A
    T = -4*sparse.eye(m,m,k=0) + sparse.eye(m,m,k=1) + sparse.eye(m,m,k=-1)
    I = sparse.eye(m)
    II = sparse.eye(m,k=1)+sparse.eye(m,k=-1)
    A = sparse.kron(I,T)+sparse.kron(II,I)
    A /= h**2

    # build RHS
    F = f(x[1:m+1,None],x[None,1:m+1])
    
    # subtract stencil evaluated at known boundary points
    F -= (bd[0:m,1:m+1] + bd[2:m+2,1:m+1] + bd[1:m+1,0:m] + bd[1:m+1,2:m+2])/h**2
    
    # evaluate f along grid, account for known boundary points, and flatten
    F_ = np.reshape(F,(-1,))
    
    # solve system
    U_ = sparse.linalg.spsolve(A.tocsr(),F_)
    
    # construct 2D solution
    U = bd
    U[1:m+1,1:m+1] = U_.reshape(m,m)
    
    # plot output
    if(img_q):
        plt.figure()
        plt.pcolormesh(x,x,U.T) # transpose because x are rows
        plt.axis('image')
        plt.colorbar()
        plt.savefig('img/1/'+str(m)+'.pdf')
    
    return U
#<endTeX>
def compare_mesh():
    E = 9
    M = 2**E-1
    fine_solution = five_point_poisson(M,True)

    global h,error_inf,error_2
    h, error_inf, error_2 = [], [], []
    
    for e in [2,3,4,5,6,7,8]:
        m = 2**e-1
        coarse_solution = five_point_poisson(m,True)
        diff = np.abs(coarse_solution - fine_solution[::2**(E-e),::2**(E-e)])
        h = np.append(h,1/(m+1))
        error_inf = np.append(error_inf,np.max(diff))
        error_2 = np.append(error_2,h[-1]*np.linalg.norm(diff,'fro'))

def save_figures():
    linear_fit_inf = np.poly1d(np.polyfit(np.log2(h), np.log2(error_inf), 1))
    linear_fit_2 = np.poly1d(np.polyfit(np.log2(h), np.log2(error_2), 1))
    np.savetxt('img/1/fit_inf.txt',[linear_fit_inf[1]],fmt="%.6s")
    np.savetxt('img/1/fit_2.txt',[linear_fit_2[1]],fmt="%.6s")
    print(linear_fit_inf[1], linear_fit_2[1])
    
    plt.figure()
    plt.scatter(np.log2(h),np.log2(error_inf))
    plt.plot(np.log2(h), linear_fit_inf(np.log2(h)), label = 'slope = '+str(round(linear_fit_inf[1],3)))
    plt.legend()
    plt.savefig('img/1/error_inf.pdf')
    
    plt.figure()
    plt.scatter(np.log2(h),np.log2(error_2))
    plt.plot(np.log2(h), linear_fit_2(np.log2(h)), label = 'slope = '+str(round(linear_fit_2[1],3)))
    plt.legend()
    plt.savefig('img/1/error_2.pdf')
    
    table = np.array([h,error_2, error_inf],dtype='<U32').T
    np.savetxt('5pp_error.txt',table,delimiter=' & ', newline=' \\\\ \hline \n',fmt='%15s')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:46:05 2018

@author: tyler
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)


a = 0.5

def phi(x):
    return 20*np.pi*x**3

def dphi(x):
    return 60*np.pi*x**2

def ddphi(x):
    return 120*np.pi*x
    
def f(x):
    return -20+a*ddphi(x)*np.cos(phi(x))-a*dphi(x)**2*np.sin(phi(x))

def u_true(x):
    return 1+12*x-10*x**2+a*np.sin(phi(x))

n = 255
h = 1/(n+1)
x = np.linspace(0,1,n+2)[1:-1]
u0 = 1+2*x

A = sparse.diags([2*np.ones(n),-np.ones(n-1),-np.ones(n-1)],[0,1,-1])/h**2
b = -f(x)
b[0] += 1/h**2
b[-1] += 3/h**2

def GS(A,b,u0,max_iter=1000):
    u = u0

    N = sparse.tril(A, format='csc')
    M = sparse.triu(A,1, format='csc')

    for iter in range(max_iter):
        if iter in [0,5,10,100,1000]:
            plt.figure()
            plt.plot(x,u_true(x))
            plt.plot(x,u)
            plt.savefig('img/3/GS_sol_'+str(iter)+'.pdf')

            plt.figure()
            plt.plot(x,u_true(x)-u)
            plt.savefig('img/3/GS_diff_'+str(iter)+'.pdf')
            
            error = np.sqrt(h)*np.linalg.norm(u_true(x)-u)
            np.savetxt('img/3/GS_err_'+str(iter)+'.txt',[error],fmt='%1.4f')

        u = sparse.linalg.spsolve_triangular(N,b-M.dot(u))

iter_cg = 0
def cg(A,b,u0,max_iter=30):
    global iter_cg
    
    u = u0
    plt.figure()
    plt.plot(x,u_true(x))
    plt.plot(x,u)
    plt.savefig('img/3/cg_sol_0.pdf')
    
    plt.figure()
    plt.plot(x,u_true(x)-u)
    plt.savefig('img/3/cg_diff_0.pdf')
    
    error = np.sqrt(h)*np.linalg.norm(u_true(x)-u)
    np.savetxt('img/3/cg_err_'+str(iter_cg)+'.txt',[error],fmt='%1.4f')


    def cg_callback(u):
        global iter_cg
        if iter_cg in [-1,4,9,19]:
            plt.figure()
            plt.plot(x,u_true(x))
            plt.plot(x,u)
            plt.savefig('img/3/cg_sol_'+str(iter_cg+1)+'.pdf')
            
            plt.figure()
            plt.plot(x,u_true(x)-u)
            plt.savefig('img/3/cg_diff_'+str(iter_cg+1)+'.pdf')
            
            error = np.sqrt(h)*np.linalg.norm(u_true(x)-u)
            np.savetxt('img/3/cg_err_'+str(iter_cg+1)+'.txt',[error],fmt='%1.4f')
        iter_cg += 1
    
    sparse.linalg.cg(A,b,x0=u,maxiter=30,callback=cg_callback)

    

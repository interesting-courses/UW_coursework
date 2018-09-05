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

def Icf_(n):
    I = sparse.csr_matrix((2*n+1,n));
    for i in range(n):
        I[2*i:2*i+3,i] = np.transpose([[.5,1,.5]])
    return I

def Ifc_(n):
    return .5*Icf_(n).T

#n is number of interior points
def A_(n):
    return sparse.diags([-2*np.ones(n),np.ones(n-1),np.ones(n-1)],[0,1,-1])*(n+1)**2


    
def f_(x,a):
    def phi(x):
        return 20*np.pi*x**3
    
    def dphi(x):
        return 60*np.pi*x**2
    
    def ddphi(x):
        return 120*np.pi*x

    return -20+a*ddphi(x)*np.cos(phi(x))-a*dphi(x)**2*np.sin(phi(x))

def u_true(x,a):
    def phi(x):
        return 20*np.pi*x**3

    return 1+12*x-10*x**2+a*np.sin(phi(x)) -1- 2*x


# figure out how to solve nonzero BCs

#<startMG>
def multi_grid(A_,f_,M_,u0,J,max_iter,tol):
    ns = [len(u0)]
    for j in range(J):
        ns.append(int((ns[-1]+1)/2)-1)

    al = u0[0]
    be = u0[-1]
    def b_(f,n,al,be):
        b = f
        b[0] -= al*(n+1)**2
        b[-1] -= be*(n+1)**2
        
        return b
    
    A = list(map(A_,ns))
    M = list(map(M_,A))
    xs = list(map(lambda n:np.linspace(0,1,n+2)[1:-1],ns))
    b = b_(f_(xs[0],0.5),ns[0],al,be)
    
    Ifc = list(map(Ifc_,ns))[1:]
    Icf = list(map(Icf_,ns))
    
    r = list(map(lambda n:np.zeros(n),ns))
    f = list(map(lambda n:np.zeros(n),ns))
    delta = list(map(lambda n:np.zeros(n),ns))
    d = list(map(lambda n:np.zeros(n),ns))
    
    u=u0
    res_norm = []
    r[0] = b - A[0].dot(u)
    for k in range(max_iter):
        res_norm.append(np.linalg.norm(r[0])/np.linalg.norm(b))
        if res_norm[-1] < tol:
            return u,k          
	
	# smooth
        for j in range(1,J):
            f[j] = Ifc[j-1].dot(r[j-1])
            delta[j] = sparse.linalg.spsolve_triangular(M[j],f[j])
            r[j] = f[j] - A[j].dot(delta[j])
        
	# project
        f[J] = Ifc[J-1].dot(r[J-1])
        
        # solve on corse grid 
        d[J]= sparse.linalg.spsolve(A[J],f[J])
        
	# smooth
        for j in range(J-1,0,-1):
            delta[j] += Icf[j+1].dot(d[j+1])
            d[j] = delta[j] + sparse.linalg.spsolve_triangular(M[j],f[j]-A[j].dot(delta[j]))
        
	# interpolate  
        u += Icf[1].dot(d[1])
        
        r[0] = b - A[0].dot(u)
        u += sparse.linalg.spsolve_triangular(M[0],r[0])

    return u,-1
#<endMG>
    
def make_plot(n):
    
    x=np.linspace(0,1,n+2)[1:-1]
    u0 = 0*x
    
    def M_j(A):
        w = 3/2
        return w*sparse.diags(A.diagonal(),format='csc')

    u,k = multi_grid(A_,f_,M_j,u0,1,100,1e-10)

    plt.figure()
    plt.plot(x,u_true(x,0.5))
    plt.plot(x,u)
    plt.savefig('img/mg_2_sol.pdf')

def multi_grid_analysis():
    tol = 1e-10

    J = 6
    ns = 2**np.arange(J+1,J+5)-1

    def M_j(A):
        w = 3/2
        return w*sparse.diags(A.diagonal(),format='csc')
    def M_gs(A):
        return sparse.tril(A,format='csc')
    
    iter_j = []
    iter_gs = []
    for N in ns:
        u0 = np.zeros(N)
        iter_j.append(multi_grid(A_,f_,M_j,u0,J,100,tol)[1])
        iter_gs.append(multi_grid(A_,f_,M_gs,u0,J,100,tol)[1])
    
    plt.figure()
    plt.scatter(ns,iter_j)
    plt.savefig('img/multigrid_j_'+str(J)+'.pdf')
    
    plt.figure()
    plt.scatter(ns,iter_gs)
    plt.savefig('img/multigrid_gs_'+str(J)+'.pdf')


def two_grid(A_,f_,al,be,N,max_iter,tol):
    n = int((N-1)/2)
    
    xc = np.linspace(0,1,n+2)[1:-1]
    xf = np.linspace(0,1,2*n+1+2)[1:-1]
    
    Ac = A_(n)
    Af = A_(2*n+1)
    
    def b_(f,n,al,be):
        b = f
        b[0] -= al*(n+1)**2
        b[-1] -= be*(n+1)**2
        
        return b
        

    bf = b_(f_(xf,0.5),n,0,0)
    
    Icf = Icf_(n)
    Ifc = Ifc_(n)
    
    # Gauss-Seidel
    Mc = sparse.tril(Ac,0,format='csc')
    Mf = sparse.tril(Af,0,format='csc')

    # jacobi
    w = 3/2
    Mc = w*sparse.diags(Ac.diagonal(),format='csc')
    Mf = w*sparse.diags(Af.diagonal(),format='csc')
    
    res_norm = [] 
    uf = 0*xf
    rf = bf-Af.dot(uf)
    for k in range(max_iter):
        
        res_norm.append(np.linalg.norm(rf)/np.linalg.norm(bf))
        if res_norm[-1] < tol:
            return uf,k
        
        # project to coarse grid
        rc = Ifc.dot(rf)
        
        # solve on coarse grid
        zc = sparse.linalg.spsolve(Ac,rc)
        
        # interpolate to fine grid and update fine solution
        uf += Icf.dot(zc)
        
        # smooth on fine grid
        rf = bf-Af.dot(uf)
        uf += sparse.linalg.spsolve_triangular(Mf,rf)
    
    return uf,-1

def two_step_plot(n):
    tol = 1e-10
    uf,iter = two_grid(A_,f_,1,3,n,100,tol)
    

    xf = np.linspace(0,1,n+2)[1:-1]
    plt.plot(xf,uf)
    plt.plot(xf,u_true(xf,0.5))

def two_grid_analysis():
    tol = 1e-10
    iter = []
    ns = 100*np.arange(3,30,5)+1
    for N in ns:
        iter.append(two_grid(A_,f_,0,0,N,100,tol)[1])
        
    plt.scatter(ns,iter)

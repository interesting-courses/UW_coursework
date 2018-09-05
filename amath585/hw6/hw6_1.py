1#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)

#  Solves the steady-state heat equation in a square with conductivity
#  c(x,y) = 1 + x^2 + y^2:
#
#     -d/dx( (1+x^2+y^2) du/dx ) - d/dy( (1+x^2+y^2) du/dy ) = f(x),   
#                                                       0 < x,y < 1
#     u(x,0) = u(x,1) = u(0,y) = u(1,y) = 0
#
#  Uses a centered finite difference method.

# return A and b to solve steady state heat equation    
def Ab(n):
    #  Set up grid.
    #n = int(input(' Enter number of subintervals in each direction: '));
    h = 1/n
    N = (n-1)**2

    # Form block tridiagonal finite difference matrix A and right-hand side 
    # vector b.
    A = sparse.csr_matrix((N,N));
    b = np.ones((N,1));         # Use right-hand side vector of all 1's.

    # Loop over grid points in y direction.
    for j in range (n-1):
        yj = (j+1)*h
        yjph = yj+h/2;  yjmh = yj-h/2
    
        # Loop over grid points in x direction.
        for i in range(n-1):
            xi = (i+1)*h
            xiph = xi+h/2;  ximh = xi-h/2
            aiphj = 1 + xiph**2 + yj**2
            aimhj = 1 + ximh**2 + yj**2
            aijph = 1 + xi**2 + yjph**2
            aijmh = 1 + xi**2 + yjmh**2
            k = (j)*(n-1) + i
            A[k,k] = aiphj+aimhj+aijph+aijmh
            if i > 0: A[k,k-1] = -aimhj
            if i < n-2: A[k,k+1] = -aiphj
            if j > 0: A[k,k-(n-1)] = -aijmh
            if j < n-2: A[k,k+(n-1)] = -aijph
    
    return (A/h**2,b)   # Remember to multiply A by (1/h^2).

#n = int(input(' Enter number of subintervals in each direction: '));
def direct_solve():
    n = 8
    A,b = Ab(n)
    
    # Solve linear system.
    u_comp = sparse.linalg.spsolve(A,b)

#<startJacobi>
def jacobi(A,b,tol,max_iter=20):
    n = np.shape(A)[1]
    x = sparse.csc_matrix((n,1))
    
    M = sparse.diags(A.diagonal(), format='csc')

    residual_norm=np.zeros(max_iter)
    iter_tol = -1
    for iter in range(max_iter):
        residual = b-A.dot(x)
        x += sparse.linalg.spsolve_triangular(M,residual)
        residual_norm[iter] = np.linalg.norm(residual)/np.linalg.norm(b)
        if residual_norm[iter] < tol: 
            iter_tol = iter
            break

    return (x,residual_norm,iter_tol)      
#<endJacobi>

#<startGS>
def gauss_seidel(A,b,tol,max_iter=20):
    n = np.shape(A)[1]
    x = sparse.csc_matrix((n,1))
    
    M = sparse.tril(A,0, format='csc')

    residual_norm=np.zeros(max_iter)
    iter_tol = -1
    for iter in range(max_iter):
        residual = b-A.dot(x)
        x += sparse.linalg.spsolve_triangular(M,residual)
        residual_norm[iter] = np.linalg.norm(residual)/np.linalg.norm(b)
        if residual_norm[iter] < tol: 
            iter_tol = iter
            break

    return (x,residual_norm,iter_tol)      
#<endGS>
    
#<startSOR>
def SOR(A,b,w,tol,max_iter=20,opt=False):
    n = np.shape(A)[1]
    x = sparse.csc_matrix((n,1))
    
    D = sparse.diags(A.diagonal(), format='csc')
    L = sparse.tril(A,-1, format='csc')
    M = D/w+L

    if opt:
        G = sparse.eye(n) - sparse.diags(1/A.diagonal(),format='csc').dot(A)
        p = np.linalg.norm(sparse.linalg.eigs(G,1,which='LM')[0])
        w = 2/(1+np.sqrt(1-p**2))
        print(w)
    residual_norm=np.zeros(max_iter)
    
    iter_tol = -1
    for iter in range(max_iter):
        residual = b-A.dot(x)
        x += sparse.linalg.spsolve_triangular(M,residual)
        residual_norm[iter] = np.linalg.norm(residual)/np.linalg.norm(b)
        if residual_norm[iter] < tol: 
            iter_tol = iter
            break

    return (x,residual_norm,iter_tol)
#<endSOR>

def tests():
    n = 20
    A,b = Ab(n)

    max_iter = 175
    tol = 1e-15
    res_jacobi = jacobi(A,b,tol,max_iter)
    res_GS = gauss_seidel(A,b,tol,max_iter)
    res_SOR3 = SOR(A,b,1.3,tol,max_iter)
    res_SOR5 = SOR(A,b,1.5,tol,max_iter)
    res_SOR7 = SOR(A,b,1.7,tol,max_iter)
#    res_SOR_opt = SOR(A,b,1,tol,max_iter,opt=True)
    
    plt.yscale('log')
    plt.plot(range(len(res_jacobi[1])),res_jacobi[1])
    plt.plot(range(len(res_GS[1])),res_GS[1])
    plt.plot(range(len(res_SOR3[1])),res_SOR3[1])
    plt.plot(range(len(res_SOR5[1])),res_SOR5[1])
    plt.plot(range(len(res_SOR7[1])),res_SOR7[1])
#    plt.plot(range(len(res_SOR_opt[1])),res_SOR_opt[1])
    plt.savefig('img/1/iter_convergence_'+str(n)+'.pdf')



def CG_tests():
    n = 20
    A,b = Ab(n)
    residual_CG_ichol = []
    residual_CG = []
   
    def CG_callback(x):
        nonlocal residual_CG
        residual_CG = np.append(residual_CG,np.linalg.norm(b-A.dot(x))/np.linalg.norm(b))
    
    def CG_callback_ichol(x):
        nonlocal residual_CG_ichol
        residual_CG_ichol = np.append(residual_CG_ichol,np.linalg.norm(b-A.dot(x))/np.linalg.norm(b))
    
    max_iter = 300
        
    tol = 1e-15
    N = (n-1)**2
    
#<startCG>
    lu = sparse.linalg.spilu(A,drop_tol = 1e-3)
    M = lu.solve(np.identity(N))
#<endCG>
    
    sparse.linalg.cg(A,b,x0=np.zeros((n-1)**2),tol=tol,maxiter=max_iter,callback=CG_callback)
    sparse.linalg.cg(A,b,x0=np.zeros((n-1)**2),tol=tol,maxiter=max_iter,M=M,callback=CG_callback_ichol)

    plt.yscale('log')
    plt.plot(range(len(residual_CG)),residual_CG)
    plt.plot(range(len(residual_CG_ichol)),residual_CG_ichol)
    plt.savefig('img/1/cg_convergence_'+str(n)+'.pdf')
    
def iter_vs_h():
    iterj = []
    iter0 = []
    iter3 = []
    iter5 = []
    iter7 = []
    h = []
    tol = 1e-5
    max_iter = 2000
    
    for n in [3,4,5,7,10,14,20,30]:
        A,b = Ab(n)
        h = np.append(h,1/(n+1))
        iterj = np.append(iterj,jacobi(A,b,tol,max_iter=max_iter)[2])
        iter0 = np.append(iter0,SOR(A,b,1,tol,max_iter=max_iter)[2])
        iter3 = np.append(iter3,SOR(A,b,1.3,tol,max_iter=max_iter)[2])
        iter5 = np.append(iter5,SOR(A,b,1.5,tol,max_iter=max_iter)[2])
        iter7 = np.append(iter7,SOR(A,b,1.7,tol,max_iter=max_iter)[2])

    plt.yscale('log')
    plt.plot(h,iterj)    
    plt.plot(h,iter0)
    plt.plot(h,iter3)
    plt.plot(h,iter5)
    plt.plot(h,iter7)
    plt.savefig('img/1/iter_vs_h.pdf')


def iter_vs_h_cg():
    global iter_cg_temp
    global iter_pcg_temp
    iter_cg_temp = 0
    iter_pcg_temp = 0

    def CG_callback(x):
        global iter_cg_temp
        iter_cg_temp += 1
    
    def PCG_callback(x):
        global iter_pcg_temp
        iter_pcg_temp += 1
        
    iter_cg = []
    iter_pcg = []

    h = []
    tol = 1e-16
    max_iter = 2000
    
    for n in [3,5,10,20,30]:
        iter_cg_temp = 0
        iter_pcg_temp = 0
        
        N = (n-1)**2
        
        A,b = Ab(n)
        h = np.append(h,1/(n+1))
        lu = sparse.linalg.spilu(A)
        Pr = sparse.csc_matrix((N, N))
        Pr[lu.perm_r, np.arange(N)] = 1
        Pc = sparse.csc_matrix((N, N))
        Pc[np.arange(N), lu.perm_c] = 1
    
        M =  Pc.dot((sparse.linalg.inv(lu.U).dot( sparse.linalg.inv(lu.L)) ).dot(Pr))
    
        sparse.linalg.cg(A,b,x0=np.zeros((n-1)**2),tol=tol,maxiter=max_iter,callback=CG_callback)
        sparse.linalg.cg(A,b,x0=np.zeros((n-1)**2),tol=tol,maxiter=max_iter,M=M,callback=PCG_callback)
                
        iter_cg = np.append(iter_cg,iter_cg_temp)
        iter_pcg = np.append(iter_pcg,iter_pcg_temp)

    plt.plot(h,iter_cg)    
    plt.plot(h,iter_pcg)
    plt.savefig('img/1/iter_vs_h_cg.pdf')
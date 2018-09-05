#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)
from scipy import integrate

#<start:RK>
def r_k(f,u0,N,T):
    k = T/N
    U = np.zeros((len(u0),N))
    U[:,0] = u0
    for n in range(N-1):
        tn = k*n
        Y1 = U[:,n]
        Y2 = U[:,n] + k/2*f(tn,Y1)
        Y3 = U[:,n] + k/2*f(tn+k/2,Y2)
        Y4 = U[:,n] + k*f(tn+k/2,Y3)
        U[:,n+1] = U[:,n] + k/6*( f(tn,Y1) + 2*f(tn+k/2,Y2) + 2*f(tn+k/2,Y3) + f(tn+k,Y4))
    return U
#<end:RK>

# forward Euler
#<start:fw>
def fw_euler(f,u0,N,T):
    U = np.zeros((len(u0),N))
    U[:,0] = u0
    for i in range(N-1):
        U[:,i+1] = U[:,i] + k*f(U[:,i])
    return U
#<end:fw>
    
# backward Euler
#<start:bw>
def bw_euler(f,u0,N,T,Jf,tol,max_iter):
    U = np.zeros((2,N))
    U[:,0] = u0
    for n in range(N-1):
        # run Newton's method with initial guess from foward Euler
        U[:,n+1] = U[:,n] + k*f(U[:,n])
        for i in range(max_iter):
            dw = np.linalg.solve(np.identity(2) - k*Jf(U[:,n+1]),U[:,n+1]-k*f(U[:,n+1])-U[:,n])
            if np.linalg.norm(dw) < tol:
                break
            U[:,n+1] -= dw
        print("max iteration reached, Newton's method did not converge to specified tolerance")
    return U
#<end:bw>

T = 400
eps = 1/100
u0 = [2,2/3]
def f(u):
    x = u[0]
    a = u[1]
    return np.array([-x**3/3+x+a, -eps*x])

def Jf(u):
    x = u[0]
    a = u[1]
    return np.array([[-x**2+1, 1],[-eps, 0]])

# RK45
sol = integrate.solve_ivp(lambda t,x: f(x),[0,T],u0,tol=1e-5)

for N in [300, 350, 600]:
    k = T/N
    
    U_fe = fw_euler(f,u0,N,T)
    U_rk = r_k(lambda t,u: f(u),u0,N,T)
    U_be = bw_euler(f,u0,N,T,Jf,1e-16,100)

    t = np.linspace(0,T,N)
    plt.figure()
    plt.plot(sol.t,sol.y[0],color='.8')
    plt.plot(sol.t,sol.y[1],color='.8',linestyle='--')
    plt.plot(t,U_fe[0],color='0')
    plt.plot(t,U_fe[1],color='0',linestyle='--')
    plt.savefig('img/fe_'+str(N)+'.pdf')

    plt.figure()
    plt.plot(sol.t,sol.y[0],color='.8')
    plt.plot(sol.t,sol.y[1],color='.8',linestyle='--')
    plt.plot(t,U_rk[0],color='0')
    plt.plot(t,U_rk[1],color='0',linestyle='--')
    plt.savefig('img/rk_'+str(N)+'.pdf')
    
    plt.figure()
    plt.plot(sol.t,sol.y[0],color='.8')
    plt.plot(sol.t,sol.y[1],color='.8',linestyle='--')
    plt.plot(t,U_be[0],color='0')
    plt.plot(t,U_be[1],color='0',linestyle='--')
    plt.savefig('img/be_'+str(N)+'.pdf')
    

print(len(sol.t))
plt.figure()
plt.plot(sol.t,sol.y[0],color='0')
plt.plot(sol.t,sol.y[1],color='0',linestyle='--')
plt.savefig('img/rk45.pdf')
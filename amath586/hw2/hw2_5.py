#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)
from scipy import integrate
from scipy import io

#%%
def f(t,u):
    return np.array([-1000*u[0] + u[1], -u[1]/10])


#<startRK>
def r_k(f,u0,N,T):
    k = T/N
    u = np.zeros((len(u0),N+1))
    u[:,0] = u0
    for n in range(N):
        tn = k*n
        Y1 = u[:,n]
        Y2 = u[:,n] + k/2*f(tn,Y1)
        Y3 = u[:,n] + k/2*f(tn+k/2,Y2)
        Y4 = u[:,n] + k*f(tn+k/2,Y3)
        u[:,n+1] = u[:,n] + k/6*( f(tn,Y1) + 2*f(tn+k/2,Y2) + 2*f(tn+k/2,Y3) + f(tn+k,Y4))
    return u
#<endRK>

#%%
T = 1
u0 = [1,2]
error_rk = []
Ns = np.array([51,375,500,1200])
for N in Ns:
    t = np.linspace(0,T,N+1)
    u_true = np.array([9979/9999*np.exp(-1000*t)+20/9999*np.exp(-t/10),2*np.exp(-t/10)])
    u_rk = r_k(f,u0,N,T)

    error_rk.append(np.linalg.norm(u_true-u_rk,np.inf))

    plt.figure()
    ax = plt.gca()
    ax.plot(t,u_true[0],color='.7')
    ax.plot(t,u_rk[0],color='.1',marker='o',ms=3,linestyle='None')

    ax.plot(t,u_true[1],color='.7')
    ax.plot(t,u_rk[1],color='.1',marker='s',ms=3,linestyle='None')

#        ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig('img/5/N'+str(N)+'.pdf')


#%%
sol = integrate.solve_ivp(f,[0,1],[1,2],method='RK23')

t_23 = sol.t
u_23 = sol.y
u_true = np.array([9979/9999*np.exp(-1000*t_23)+20/9999*np.exp(-t_23/10),2*np.exp(-t_23/10)])
error_23 = np.linalg.norm(u_true-u_23,np.inf)

ax = plt.gca()
ax.plot(t_23,u_true[0],color='.5')
ax.plot(t_23,u_23[0],color='.1',marker='o',ms=3,linestyle='None')

ax.plot(t_23,u_true[1],color='.5')
ax.plot(t_23,u_23[1],color='.1',marker='s',ms=3,linestyle='None')

#        ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig('img/5/RK23.pdf')

plt.figure()
ax = plt.gca()
ax.plot(np.linspace(0,1,len(t_23)),t_23,'k')
plt.savefig('img/5/RK23_t.pdf')

#%%

tu = io.loadmat('tu.mat')
t_23s = tu['t'][:,0]
u_23s = tu['u'].T

u_true = np.array([9979/9999*np.exp(-1000*t_23s)+20/9999*np.exp(-t_23s/10),2*np.exp(-t_23s/10)])
error_23s = np.linalg.norm(u_true-u_23s,np.inf)

ax = plt.gca()
ax.plot(t_23s,u_true[0],color='.5')
ax.plot(t_23s,u_23s[0],color='.1',marker='o',ms=3,linestyle='None')

ax.plot(t_23s,u_true[1],color='.5')
ax.plot(t_23s,u_23s[1],color='.1',marker='s',ms=3,linestyle='None')

#        ax.set_xscale('log')
ax.set_yscale('log')

plt.savefig('img/5/ode23s.pdf')

plt.figure()
ax = plt.gca()
ax.plot(np.linspace(0,1,len(t_23s)),t_23s,'k')
plt.savefig('img/5/ode23s_t.pdf')

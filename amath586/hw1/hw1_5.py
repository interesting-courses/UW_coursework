#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)

def f(u,t):
    return u**2 - np.sin(t) - np.cos(t)**2

#<startEuler>
def fw_euler(f,u0,N,T):
    k = T/N
    u = np.zeros(N+1)
    u[0] = u0
    for n in range(N):
        u[n+1] = u[n] + k*f(u[n],k*n)
    return u
#<endEuler>

#<startRK>
def r_k(f,u0,N,T):
    k = T/N
    u = np.zeros(N+1)
    u[0] = u0
    for n in range(N):
        tn = k*n
        Y1 = u[n]
        Y2 = u[n] + k/2*f(Y1,tn)
        Y3 = u[n] + k/2*f(Y2,tn+k/2)
        Y4 = u[n] + k*f(Y3,tn+k/2)
        u[n+1] = u[n] + k/6*( f(Y1,tn) + 2*f(Y2,tn+k/2) + 2*f(Y3,tn+k/2) + f(Y4,tn+k))
    return u
#<endRK>

def make_plots():
    error_e = []
    error_rk = []
    Ns = np.array([25,50,100,200,400,800,1600])
    for N in Ns:
        T = 8
        x = np.linspace(0,T,N+1)
        u_true = np.cos(x)
        u_e = fw_euler(f,1,N,T)
        u_rk = r_k(f,1,N,T)

        error_e.append(np.linalg.norm(u_true-u_e,np.inf))
        error_rk.append(np.linalg.norm(u_true-u_rk,np.inf))

        if N==25:
            plt.figure()
            plt.plot(x,u_true,color='.5')
            plt.plot(x,u_e,color='.1',marker='o',ms=4,linestyle='None')
            plt.plot(x,u_rk,color='.1',marker='s',ms=4,linestyle='None')
            plt.savefig('img/5/N25.pdf')


    ks = 8/Ns
    coeff_e = np.polyfit(np.log10(ks), np.log10(error_e), 1)
    fit_e = np.poly1d(coeff_e)

    coeff_rk = np.polyfit(np.log10(ks), np.log10(error_rk), 1)
    fit_rk = np.poly1d(coeff_rk)

    np.savetxt('img/5/coeff_e.txt',[coeff_e[0]],fmt='%.2f',delimiter='',newline='')
    np.savetxt('img/5/coeff_rk.txt',[coeff_rk[0]],fmt='%.2f',delimiter='',newline='')

    print(coeff_e,coeff_rk)

    fig = plt.figure()
    ax = plt.gca()
    plt.plot(ks,10**(fit_e(np.log10(ks))),'.5')
    plt.plot(ks,10**(fit_rk(np.log10(ks))),'.5')
    ax.plot(ks,error_e,color='k',marker='o',ms=4,linestyle='None')
    ax.plot(ks,error_rk,color='k',marker='s',ms=4,linestyle='None')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.savefig('img/5/error.pdf')

make_plots()

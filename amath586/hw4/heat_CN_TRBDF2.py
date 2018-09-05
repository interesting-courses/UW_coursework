#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 21:11:44 2018

@author: tyler
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

#<start>
def heat_CN_TRBDF2(m):
#
# heat_CN.py
#
# Solve u_t = kappa * u_{xx} on [ax,bx] with Dirichlet boundary conditions,
# using the Crank-Nicolson method with m interior points.
#
# Returns k, h, and the max-norm of the error.
# This routine can be embedded in a loop on m to test the accuracy,
# perhaps with calls to error_table and/or error_loglog.
#
# Original MATLAB code from  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)
# Ported to Python by Tyler Chen (2018)
    
    plt.figure()              # clear graphics
                              # Put all plots on the same graph (comment out if desired)
    
    ax = 0;
    bx = 1;
    kappa = .02;                 # heat conduction coefficient:
    tfinal = 1;                  # final time
    
    h = (bx-ax)/(m+1);           # h = delta x
    x = np.linspace(ax,bx,m+2);  # note x(1)=0 and x(m+2)=1
                                 # u(1)=g0 and u(m+2)=g1 are known from BC's
    k = 4*h;                     # time step
    
    nsteps = round(tfinal / k);  # number of time steps

    #nplot = 1;      # plot solution every nplot time steps
                     # (set nplot=2 to plot every 2 time steps, etc.)
    nplot = nsteps;  # only plot at final time
    
    if abs(k*nsteps - tfinal) > 1e-5:
        # The last step won't go exactly to tfinal.
        print(' ')
        print('WARNING *** k does not divide tfinal, k = %1.5f' % k)
        print(' ')
        
    # true solution for comparison:
    # For Gaussian initial conditions u(x,0) = exp(-beta * (x-0.4)^2)
    beta = 150;
    utrue = lambda x,t: np.exp(-(x-0.4)**2 / (4*kappa*t + 1/beta)) / np.sqrt(4*beta*kappa*t+1);
    
    # initial conditions:
    u0 = utrue(x,0);
    
    # Each time step we solve MOL system U' = AU + g using the TRBDF2
    
    # set up matrices:
    r = kappa * k/(h**2);
    e = np.ones(m);
    A = sparse.spdiags([e,-2*e,e],[-1,0,1],m,m)
    A1_ = sparse.eye(m) + (r / 4) * A;
    A2_ = sparse.eye(m) - (r / 4) * A;
    A2 = sparse.eye(m) - (r / 3) * A;

    # initial data on fine grid for plotting:
    xfine = np.linspace(ax,bx,1001);
    ufine = utrue(xfine,0);

    # initialize u and plot:
    tn = 0;
    u = u0;
    
    plt.plot(x,u,'b.-', xfine,ufine,'r')
    plt.legend(['computed','true'])
    plt.title('Initial data at time = 0')


    # main time-stepping loop:
    for n in range(nsteps):
        tnp = tn + k;   # = t_{n+1}
        # boundary values u(0,t) and u(1,t) at times tn and tnp:
        
        g0n = u[0];
        g1n = u[m+1];
        g0n_ = utrue(ax,tn+k/2);
        g1n_ = utrue(bx,tn+k/2);
        g0np = utrue(ax,tnp);
        g1np = utrue(bx,tnp);

        # compute right hand side for intermediate linear system:
        uint = u[1:-1];   # interior points (unknowns)
        rhs_ = A1_ @ uint;
        # fix-up right hand side using BC's (i.e. add vector g to A2*uint)
        rhs_[0] += (r / 4) * (g0n + g0n_);
        rhs_[m-1] += (r / 4) * (g1n + g1n_);
        
        # solve intermediate linear system:
        uint_ = sparse.linalg.spsolve(A2_,rhs_);
        
        # compute right hand side for linear system:
        rhs = (4 * uint_ - uint) / 3
        rhs[0] += (r / 3) * g0np;
        rhs[m-1] += (r / 3) * g1np;
        
        # solve linear system:
        uint = sparse.linalg.spsolve(A2,rhs)
        
        # augment with boundary values:
        u = np.concatenate([[g0np], uint, [g1np]]);
        # plot results at desired times:
        if (n+1)%nplot==0 or (n+1)==nsteps:
            print(n)
            ufine = utrue(xfine,tnp);
            plt.plot(x,u,'b.-', xfine,ufine,'r')
            plt.title('t = %1.5f  after %i time steps with %i grid points' % (tnp,n+1,m+2))
            error = max(abs(u-utrue(x,tnp)));
            print('at time t = %.5f  max error =  %.5f'%(tnp,error))
            if (n+1)<nsteps: input('Hit <return> to continue  ')
    
        tn = tnp;   # for next time step
    plt.show()
    
    return k,h,error
#<end>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)


def rhs_dyn(t,x,mu):
    return [x[1], mu*(1-x[0]**2)*x[1]-x[0]];

dt=0.01;
t=np.arange(0,50,dt);
x0=[0.1, 5];
mu=1.2;

sol = solve_ivp(lambda t,x: rhs_dyn(t,x,mu),[0,50],x0,method='RK45',t_eval=t)
#<endTeX>

t = sol.t
[x0,x1] = sol.y

x0dot = (x0[2:] - x0[:-2]) / (2*dt)
x1dot = (x1[2:] - x1[:-2]) / (2*dt)

x0s=x1[1:-1];
x1s=x2[1:-1];

np.shape(E_)
np
# make exponent list
max_deg = 3
E_ = [[(m,n) for m in range(max_deg)] for n in range(max_deg)]
E = np.reshape(E_,(max_deg**2,2))
A = np.array([x0s**m * x1s**n for (m,n) in E]).T
#A = np.array([x1s, x2s, x1s**2, x1s*x2s, x2s**2, x1s**3, (x1s**2)*x2s, (x2s**2)*x1s, x2s**3]).T;
E
c0 = np.linalg.lstsq(A,x0dot)[0]
c1 = np.linalg.lstsq(A,x1dot)[0]

np.round(c0,decimals=3)
np.round(c1,decimals=3)

plt.figure();
plt.bar(np.arange(len(c0)),c0);
plt.show();
plt.figure();
plt.bar(np.arange(len(c1)),c1);
plt.show();

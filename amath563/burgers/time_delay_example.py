#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:37:43 2018

@author: tyler
"""

A = np.array([[2,3],[-1,2]])
def rhs(t,xy):
    return [xy[1]+xy[0],xy[0]+xy[1],xy[2]+xy[0]]

t = np.linspace(0,1,100)

y = integrate.solve_ivp(rhs,[0,1],[1,2,1],t_eval=t).y.T
#%%

y = np.zeros((100,3))
y[0] = [1,2,1]
for i in np.arange(0,100-1):
    y[i+1] = y[i] + 1/100 * np.array([y[i,1]+y[i,0],y[i,0]+2*y[i,1],-y[i,2]+2*y[i,0]])

#%%
plt.scatter(t,y[:,0])
plt.scatter(t,y[:,1])

d = 10
w = len(t)-d
H = np.zeros((d,w))
for i in range(d):
    H[i] = y.T[0,i:w+i]
    

# take SVD
u,s,vh = np.linalg.svd(H)

# plot singular values
plt.figure()
ax = plt.gca()
ax.set_yscale('log')
plt.plot(np.arange(d),s/np.max(s),color='0',marker='o',ms=5,linestyle='None')
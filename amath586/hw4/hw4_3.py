#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)

meshes = [19,49,99,199,499,999,1999]
hs = 1/(np.array(meshes)+1)
error = np.zeros(np.shape(meshes))

for k,m in enumerate(meshes):
    error[k] = heat_CN(m)

#%%
linear_fit = np.poly1d(np.polyfit(np.log10(hs), np.log10(np.abs(error)), 1))
    
plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.plot(hs,10**linear_fit(np.log10(hs)),color='0.8')
plt.plot(hs,np.abs(error),color='0', marker='o',linestyle='None')
plt.savefig('img/err.pdf')

np.savetxt('img/err.txt',[linear_fit[1]],fmt='%.4f')
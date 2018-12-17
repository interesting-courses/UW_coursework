#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 07:33:37 2018

@author: tyler
"""

import numpy as np

#%%
for z in []:
    cut_count = 0
    last_new_set_index = 0 # ideally this is low compared to the number of min cuts meaing we have lots of duplicate cuts
    print('dataset :',z)
    cut_sizes = np.loadtxt('b'+str(z)+'/cut_sizes_t1.dat',dtype='int')
    min_cut_size = np.min(cut_sizes)
    min_cut_locations = np.where(cut_sizes == min_cut_size)[0]
    
    cuts = set([])
    with open('b'+str(z)+'/cuts_t1.dat') as fd:
        for n, line in enumerate(fd):
            if n%5000==0:
                print(n)
            if n in min_cut_locations:
                cut_count += 1
                cut = frozenset(map(int,line.rstrip().split(',')))
                if cut not in cuts:
                    cuts.add(cut)
                    last_new_set_index = n
    
    np.savetxt('b'+str(z)+'/cut_stats.txt',[min_cut_size,len(cuts),cut_count],fmt='%d')
    print('minimum cut size found: ', min_cut_size)
    print('number of distinct min cuts found: ',len(cuts))
    print('number of min cut found: ',cut_count)

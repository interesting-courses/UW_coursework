#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import optimize

# load data from problem
G_letter = [["","C+","B","B+","A-","C",""],
     ["B-","A-","","","A+","D+","B"],
     ["B-","","B+","","A-","B","B+"],
     ["A+","","B-","A","","A-",""],
     ["","B-","D+","B+","","B","C+"]]

# convert letter grade to GPA
grade_map = {"A+":4.33,"A":4,"A-":3.66,"B+":3.33,"B":3,"B-":2.66,"C+":2.33,"C":2,"D+":1.66,"":0}

# construct grade matrix
G = np.vectorize(lambda x: grade_map[x],otypes=[float])(np.array(G_letter))

# get nonzero entries
nze = np.nonzero(G)
nnz = len(nze[0])
n,m = np.shape(G)

# construct flattened g matrix
g = G[np.nonzero(G)]

# construct coefficient matrix
C = np.zeros((nnz,n+m))
C[np.arange(nnz),nze[0]] = 1
C[np.arange(nnz),n+nze[1]] = 1

# set up linear program
A = np.block([[-C,-np.eye(nnz)],[C,-np.eye(nnz)]])
b = np.block([-g,g])
c = np.block([np.zeros(n+m),np.ones(nnz)])

# solve linear program
sol = sp.optimize.linprog(c,A,b)

# get easiness and aptitude
aptitude = sol.x[:n]
easiness = sol.x[n:n+m]

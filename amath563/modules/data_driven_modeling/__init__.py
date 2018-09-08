#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import linear_model

# FIGURE OUT HOW TO CLEAN THIS UP. WOULD PREFER TO HAVE MORE THAN ONE FOLDER..

#<start:make_tuples>
'''
Parameters
----------
l: integer
n: integer
start: integer

Returns
-------
generator of all tuples of length l whose minimum entry is start and whose entries sum to at most n
'''

def get_poly_exponents(l, n, start=0):
    if l == 0:
        yield ()
    else:
        for x in range(start, n+1):
            for t in get_poly_exponents(l - 1, n):
                if sum(t) + x <= n:
                    yield t + (x,)
#<end:make_tuples>

#<start:bin_exp>
'''
Parameters
----------
x: array like 
max_deg: integer

Returns
-------
ndarray whose entries are sums of the expansion of (x_0 + ... + x_n)^max_deg, where n = len(x). If entries of x are themselves arrays, then multiplication of terms in the expansion is done componentwise
'''

def bin_exp(x,max_deg):
    E_ = np.array(list(get_poly_exponents(len(x),max_deg)))
    E = E_[np.argsort(np.sum(E_,axis=1))]
    return np.array([np.prod([x[k]**e[k] for k in range(len(x))],axis=0) for e in E ]).T
#<end:bin_exp>
    
#<start:diff>
'''
Parameters
----------
a: array like 
dx: float
axis: integer (must be between 0 and dimension of A-1)
endpoints: boolean

Returns
-------
array numerically differentaited with mesh spacing dx a along given axis using centered second order finite difference method. If endpoints flag is True, then first order forward and backward differences are used at the endpoints. 
'''

def diff(a,dx,axis=0,endpoints=False):
    
    a = np.asanyarray(a)
    nd = a.ndim

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(2, None)
    slice2[axis] = slice(None, -2)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    
    da = np.subtract(a[slice1], a[slice2])
    if endpoints:
        slicel1 = [slice(None)] * nd
        slicel2 = [slice(None)] * nd
        slicel1[axis] = slice(1,2)
        slicel2[axis] = slice(0,1)
        slicel1 = tuple(slicel1)
        slicel2 = tuple(slicel2)
        
        dl = np.subtract(a[slicel1], a[slicel2])

        slicer1 = [slice(None)] * nd
        slicer2 = [slice(None)] * nd
        slicer1[axis] = slice(-1,None)
        slicer2[axis] = slice(-2,-1)
        slicer1 = tuple(slicer1)
        slicer2 = tuple(slicer2)
        
        dr = np.subtract(a[slicer1], a[slicer2])
        
        return np.concatenate((dl,da/2,dr),axis=axis)/dx

    else:
        return da/(2*dx)
#<end:diff>
 
#<start:pdiff>
'''
Parameters
----------
a: array like 
dx: array of float
axis: array of integer (lenth must equal dx, entries must be between 0 and dimension of A-1)
endpoints: boolean

Returns
-------
array numerically differentaited with mesh spacing dx[i] a along given axes using centered second order finite difference method in each direction. If endpoints flag is True, then first order forward and backward differences are used at the endpoints. This is a wrapper for calling diff repeatedly.
'''

def pdiff(a,dx,axis=[0],endpoints=False):
    if len(axis) == 1:
        return diff(a,dx[0],axis[0],endpoints=endpoints)
    else:
        a = diff(a,dx[0],axis[0],endpoints=endpoints)
        return pdiff(a,dx[1:],axis[1:],endpoints=endpoints)

#<end:pdiff>
       
#<start:diff2>
        '''
Parameters
----------
a: array like 
dx: float
axis: integer (must be between 0 and dimension of A-1)
endpoints: boolean

Returns
-------
array numerically differentaited with mesh spacing dx a along given axis using centered second order finite difference method for the second derivative. If endpoints flag is True, then first order forward and backward differences are used at the endpoints. Due to floating point errors it is better to use this function than diff twice, or equivalently pdiff.
'''

def diff2(a,dx,axis=0,endpoints=False):
    a = np.asanyarray(a)
    nd = a.ndim

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice3 = [slice(None)] * nd
    slice1[axis] = slice(2, None)
    slice2[axis] = slice(1,-1)
    slice3[axis] = slice(None, -2)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    slice3 = tuple(slice3)

    da = np.subtract(np.add(a[slice1], a[slice3]),2*a[slice2])

    if endpoints:
        slicel1 = [slice(None)] * nd
        slicel2 = [slice(None)] * nd
        slicel3 = [slice(None)] * nd
        slicel1[axis] = slice(2,3)
        slicel2[axis] = slice(1,2)
        slicel3[axis] = slice(0,1)
        slicel1 = tuple(slicel1)
        slicel2 = tuple(slicel2)
        slicel3 = tuple(slicel3)
        
        dl = np.subtract(np.add(a[slicel1], a[slicel3]),2*a[slicel2])

        slicer1 = [slice(None)] * nd
        slicer2 = [slice(None)] * nd
        slicer3 = [slice(None)] * nd
        slicer1[axis] = slice(-1,None)
        slicer2[axis] = slice(-2,-1)
        slicer3[axis] = slice(-3,-2)
        slicer1 = tuple(slicer1)
        slicer2 = tuple(slicer2)
        slicer3 = tuple(slicer3)
        
        dr = np.subtract(np.add(a[slicer1], a[slicer3]),2*a[slicer2])
        
        return np.concatenate((dl,da,dr),axis=axis)/dx**2

    else:
        return da/dx**2
#<end:diff2>


#wrapper for sklearn lasso to mimic matlab syntax
#<start:lasso>
'''
Parameters
----------
A: array like 
b: array like (must be compatiable with A)
alpha: float

Returns
-------
array solving minimization problem: min_x: ||b-Ax||_2 + alpha*||x||_1
'''

def lasso(A,b,alpha=1):
    reg = linear_model.Lasso(alpha=alpha/len(A))
    reg.fit(A,b)
    return reg.coef_
#<end:lasso>

#<start:kl_divergence>
'''
Parameters
----------
p: array like (normalized probability distribution)
p_m: array like (normalized probability distribution)

Returns
-------
KL divergence of distributions
'''

def kl_divergence(p,p_m):
    return np.sum(p * np.log(p / p_m))

#<start:kl_divergence>
    
#<start:AIC>
'''
Parameters
----------
model: array like (model points)
data: array like (actual data points)
k: integer (number of parameters)

Returns
-------
AIC score of models
'''

def AIC(model,data,k):
    n = len(data)
    RSS = np.linalg.norm(model-data)**2
    sigma2 = RSS / n
    logL = - n* np.log(2*np.pi) / 2 - n * np.log(sigma2) - n / 2
    return 2*k - 2*logL
#<end:AIC>

#<start:BIC>
'''
Parameters
----------
model: array like (model points)
data: array like (actual data points)
k: integer (number of parameters)

Returns
-------
BIC score of models
'''

def BIC(model,data,k):
    n = len(data)
    RSS = np.linalg.norm(model-data)**2
    sigma2 = RSS / n
    logL = - n* np.log(2*np.pi) / 2 - n * np.log(sigma2) - n / 2
    return np.log(n) * k - 2 * logL
#<end:BIC>

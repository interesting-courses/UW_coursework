import numpy as np
import matplotlib.pyplot as plt
import copy
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)


A = np.triu(np.tril(-np.ones((10,10))+3*np.identity(10),1),-1)



def power(A):
    tol=10e-16
    l=[0,1]
    v = np.random.rand(10)
    v = v/np.linalg.norm(v)
    while abs(l[-1] - l[-2]) > tol:
        w = A@v
        v = w/np.linalg.norm(w)
        l.append(v.T@A@v)
    
    plt.plot(np.log10(np.abs(l[0:len(l)-1]-l[-1])))
    plt.show()
    return [l[-1],v]


def qr(A):
    od=[]
    tol=10e-16
    A_old = np.identity(len(A))
    while abs(A[0,0] - A_old[0,0]) > tol:
        A_old=A
        [Q,R] = np.linalg.qr(A)
        A=R@Q
        od.append(np.linalg.norm(np.diagonal(A,-1)))
    print(od)
    plt.plot(np.log10(np.abs(od)))
    plt.show()
    return A
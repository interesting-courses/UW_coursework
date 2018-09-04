import numpy as np
import matplotlib 
import copy
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)


def matgen(m,condno):
    [U,X] = np.linalg.qr(np.random.randn(m,m))
    [V,X] = np.linalg.qr(np.random.randn(m,m))
    S = np.diag(condno**((1-np.linspace(1,m,m))/(m-1)))
    return U@S@V

def exercise_1():
    ge_err,inv_err,cr_err = [],[],[]
    m = 20
    for condno in [0,4,8,12,16]:
        
        A = matgen(m,10**condno)
        xtrue = np.random.rand(m)
        b = A@xtrue

        x_ge = np.linalg.solve(A,b)
        ge_err.append([condno, np.linalg.norm(x_ge-xtrue)/np.linalg.norm(xtrue),
               np.linalg.norm(b-A@x_ge)/(np.linalg.norm(A)*np.linalg.norm(x_ge))])
        
        Ainv =  np.linalg.inv(A)
        x_inv = Ainv@b
        inv_err.append([condno, np.linalg.norm(x_inv-xtrue)/np.linalg.norm(xtrue),
               np.linalg.norm(b-A@x_inv)/(np.linalg.norm(A)*np.linalg.norm(x_inv))])
    
        detA = np.linalg.det(A)
        x_cr = np.zeros(m)
        for j in range(m):
            A_j = copy.deepcopy(A)
            A_j[:,j] = b
            x_cr[j] = np.linalg.det(A_j)/detA
        cr_err.append([condno, np.linalg.norm(x_cr-xtrue)/np.linalg.norm(xtrue),
               np.linalg.norm(b-A@x_cr)/(np.linalg.norm(A)*np.linalg.norm(x_cr))])

    return [ge_err,inv_err,cr_err]

def exercise_2():
    m = 60
    A = np.tril(np.full((m,m),-1),-1)+np.identity(m)
    A[:,m-1] = np.full(m,1)
    
    x = np.random.randn(m,1)
    b = A@x
    
    x_ge = np.linalg.solve(A,b)
    
    [Q,R] = np.linalg.qr(A)
    x_qr = np.linalg.solve(R,Q.T@b)
    
#    plt.scatter(np.log10(x-x_ge),range(m))
    return [np.linalg.cond(A,2),np.linalg.norm(x-x_ge,2),np.linalg.norm(x-x_qr,2)]

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)


def exercise_2():
    out=[]
    n=50
    [U,X]=np.linalg.qr(np.random.randn(n,n))
    [V,X]=np.linalg.qr(np.random.randn(n,n))
    ss=np.sort(np.random.rand(n))
    SS = [np.diag(np.flip(ss,0)), 
          np.diag(np.flip(ss**6,0))]
    
    for i in range(2):
        S= SS[i]
        A=U@S@V
        
        [U2,S2,V2]=np.linalg.svd(A,full_matrices=True)
        S2=np.diag(S2)
        
        for j in range(n):
            if np.dot(U2[:,j],U[:,j])<0:
                U2[:,j] = -U2[:,j]
                V2[j,:] = -V2[j,:]
        res=[np.linalg.norm(U2-U),
             np.linalg.norm(V2-V),
             np.linalg.norm(S2-S)/np.linalg.norm(S),
             np.linalg.norm(A-U2@S2@V2),
             np.linalg.cond(A)]
        out.append(res)
    return out  

x=np.array([exercise_2() for i in range(50)])

for i in range(4):
    plt.scatter(x[:,0,4], np.log10(x[:,0,i]))
    plt.scatter(x[:,1,4], np.log10(x[:,1,i]))
    plt.show()
    
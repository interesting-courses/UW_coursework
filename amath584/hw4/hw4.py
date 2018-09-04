import numpy as np


def exercise_11_3():
    np.set_printoptions(precision=16)
    m,n=50,12
    
    # choose 50 linearly npaced grid points on the interval [0,1]
    ls=np.linspace(0,1,m)
    
    # evaluate 11th degree polynomial at grid points
    A=np.array([[ls[i]**j for j in range(n)] for i in range(m)])

    # evaluate function at gridpoints
    b=np.transpose(np.cos(4*ls))
    
    # solve normal equation
    At=np.transpose(A)
    parta=np.linalg.solve(At@A,At@b)
    
    # QR then solve
    Q,R=np.linalg.qr(A,mode='reduced')
    partd=np.linalg.solve(R,np.transpose(Q)@b)[0]
    
    # solve least squares directly
    parte=np.linalg.lstsq(A,b)[0]

    # SVD then solve
    U,s,Vh=np.linalg.svd(A,full_matrices=False)
    w=np.linalg.lstsq(np.diag(s),np.transpose(U)@b)[0]
    partf=np.transpose(Vh)@w
    
    print(np.transpose([parta,partd,parte,partf]))

def exercise_4_4():
    A=np.array(
            [[1,-1,0,0],
             [-1,0,1,0],
             [1,0,0,-1],
             [0,0,1,-1],
             [0,1,0,-1],
             [1,1,1,1]]
            )
    b=np.transpose(np.array([[4,9,6,3,7,20]]))
    At=np.transpose(A)
    x=np.linalg.solve(At@A,At@b)
    print(x)

exercise_11_3()
exercise_4_4()


# returns Q,R
def givens_qr(A):
    np.set_printoptions(precision=5)
    m,n=np.shape(A)
    E=np.identity(m)
    for i in range(m-1):
        v=A[i:,i]
        k=m-i
        Qk=np.identity(k)
        for j in range(1,k):
            theta=np.arctan2(v[k-j],v[k-j-1])
            Qj=np.identity(k)
            Qj[k-j-1:k-j+1,k-j-1:k-j+1]=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            v=Qj@v
            Qk=Qj@Qk
        
        Ek=np.identity(m)
        Ek[m-k:m,m-k:m]=Qk
        
        E=Ek@E
        A=Ek@A

    return(A,np.transpose(E))
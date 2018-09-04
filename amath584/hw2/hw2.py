import scipy as sp
from matplotlib import pyplot as plt

def exercise_2_4():
    #import from matlab export
    M=sp.io.loadmat('img.mat')
    X=M['X']
    
    m,n = X.shape
    
    # original matrix
    fig=plt.figure()
    plt.imshow(X,cmap='gray')
    fig.savefig('img/original.pdf',bbox_inches='tight')

    
    # get SVD
    [U,s,V] = sp.linalg.svd(X) # full_matrices=True
    S = sp.zeros((m,n))
    S[:n,:n] = sp.diag(s)
    
    for k in [509,300,150,100,50,30,20,10,5,1,0]:
        #plot rank k approximation
        fig=plt.figure()
        plt.imshow(sp.dot(U[:,:k],sp.dot(S[:k,:k],V[:k])),cmap='gray')
        fig.savefig('img/'+str(k)+'.pdf',bbox_inches='tight')
        
exercise_2_4()
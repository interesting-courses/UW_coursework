import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)

#<startTeX>
# problem discritization
def G(U):
    Gout = np.zeros(m+2)
    for i in range(1,m):
        Gout[i] = eps*(U[i-1]-2*U[i]+U[i+1])/h2+U[i]*((U[i+1]-U[i-1])/(2*h)-1)
    return Gout[1:m+1]

def J(U):
    Jout = np.zeros((m+2,m+2))
    for i in range(1,m+1):
        Jout[i,i-1] = eps/h2-2*U[i]/(2*h)
        Jout[i,i] = -2*eps/h2+(U[i+1]-U[i-1])/(2*h)-1
        Jout[i,i+1] = eps/h2+2*U[i]/(2*h)
    return Jout[1:m+1,1:m+1]

# problem parameters
alpha = -1 
beta = 1.5
eps = 0.01

a = 0
b = 1

# discritization parameters
for e in [1,2,3,4,5,6,8]:
    m = 10*2**e-1
    x = np.linspace(a,b,m+2)
    h = (b-a)/(m+1)
    h2 = h**2
            
    # initial guess
    xb = (a+b-alpha-beta)/2
    w0 = (a-b+beta-alpha)/2
    U = x - xb + w0*np.tanh(w0*(x-xb)/(2*eps))
    
    # Newton's method    
    for k in range(25):
        print(k)
        delta = np.linalg.solve(J(U),-G(U))
        U[1:m+1] += delta
    
        print(max(abs(delta)))
        if max(abs(delta)) < 10e-14:
            break
        
    plt.figure()
    plt.scatter(x,U)
    plt.savefig('img/4/'+str(m+1)+'.pdf')
#<endTeX>

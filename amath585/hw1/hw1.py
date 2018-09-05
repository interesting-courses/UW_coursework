import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)


def exercise_1():
    h = 10**np.linspace(-1,-16,16)
    x = np.pi/6
    ans = -0.5 # should i use this or sin(x)
    
    fdq = (np.sin(x+h) + np.sin(x-h) - 2*np.sin(x))/h**2
    res = np.array([np.log10(h),fdq,fdq-ans])
        
    plt.scatter(res[0],np.log10(np.abs(res[2])))
            
    return np.transpose(res)

def exercise_2():
    def fdq(u,x,h):
        return (u(x+h) + u(x-h) - 2*u(x))/h**2
    
    x = np.pi/6
    ans = -0.5
    
    phi0 = fdq(np.sin,x,np.array([0.2,0.1,0.05]))
    phi1 = [(4*phi0[1]-phi0[0])/3, (4*phi0[2]-phi0[1])/3]
    phi2 = (16*phi1[1]-phi1[0])/15
   
    out = np.append(phi0,np.append(phi1,phi2))
    print(out)
    print(out-ans)

def exercise_6():
    for m in [10,100,1000,10000]:
        h = 1/m
        h2 = h/2
        
        x = np.linspace(0,1,m+1)
        f = 2*(3*x[1:]**2-2*x[1:]+1) # start at x[1] since we evaluate for j=1,2,...
        
        f[0] -= (1+(x[1]-h2)**2)/h**2
        
        
        A = np.zeros((m,m))
        A[0,0:2] = [-2*(1+x[2]**2+h2**2),(1+(x[2]+h2)**2)]
        for j in range(2,m):
            A[j-1,j-2:j+1] = [(1+(x[j]-h2)**2),-2*(1+x[j]**2+h2**2),(1+(x[j]+h2)**2)]
        A[m-1,m-2:m] = [2*(1+x[j]**2+h2**2),-2*(1+x[j]**2+h2**2)]
    
        u = np.linalg.solve(A/h**2,f)
        print(h,np.sqrt(h)*np.linalg.norm(u-(1-x[1:])**2))

        #plt.scatter(x[1:],((1-x[1:])**2)) #actual solution
        #plt.scatter(x[1:],u) # numerical solution
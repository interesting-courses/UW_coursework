import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)

#<startTeX>
# define grid mesh types
grids = {"uniform": lambda m: np.linspace(0,1,m+2),
         "non-uniform": lambda m: (np.arange(0,m+2)/(m+1))**2
         }

# problem parameters
def rhs(x): return 2*(3*x**2-x+1)
a = 0
b = 0

def P(x): return x+x**3/3
for grid_type in ["uniform", "non-uniform"]:
    error = np.zeros((0,3))
    for e in [1,2,3,4]:

        # generate mesh
        m = 10**e-1    
        x = grids[grid_type](m) 
        d = x[1:m+2]-x[0:m+1]
        d2 = d**2
        h = np.max(d)
        xmid = (x[1:]+x[:m+1])/2
        dxmid = xmid-x[0:m+1]
        
        # generate A
        A = np.zeros((m,m))
        off_diag = - (P(x[1:m+1])-P(x[0:m])) / d2[0:m]
        main_diag = (P(x[1:m+1])-P(x[0:m])) / d2[0:m]
        main_diag += (P(x[2:m+2])-P(x[1:m+1])) / d2[1:m+1]
        
        A += np.diag(main_diag)
        A += np.diag(off_diag[1:],1) + np.diag(off_diag[1:],-1)
        
        # generate f with midpoint rule 
        fmid = rhs(xmid)        
        f = fmid[0:m]*(dxmid[0:m]) + fmid[1:m+1]*(dxmid[1:m+1])
        
        f[0] += a * (P(x[1])-P(x[0])) / (x[1]-x[0])**2
        f[m-1] += b * (P(x[m+1])-P(x[m])) / (x[m+1]-x[m])**2

        #solve
        U = np.zeros(m+2)
        U[1:m+1] = np.linalg.solve(A,f)
        U[0] = a
        U[m+1] = b
        
        error = np.append(error,[[h, np.sqrt(np.sum((U - x*(1-x))[1:m+2]*d[0:m+1])**2), np.max(np.abs( U - x*(1-x) ))]],axis=0)
        
        plt.figure()
        plt.scatter(x,x*(1-x))
        plt.scatter(x,U)
        plt.savefig('img/1/'+grid_type+'_'+str(m+1)+'.pdf')
    
    np.savetxt(grid_type+'.txt',error,delimiter=' & ', newline=' \\\\ \hline \n')
#<endTeX>

        # analytic solution
'''
def F(x): return 2*x**3-x**2+2*x
def G(x): return 3*x**4/2-2*x**3/3+x**2

f = ( G(x[1:m+1])-G(x[0:m]) - x[0:m]*(F(x[1:m+1])-F(x[0:m])) ) / d[0:m]
f += ( x[2:m+2]*(F(x[2:m+2])-F(x[1:m+1])) - (G(x[2:m+2])-G(x[1:m+1])) ) / d[1:m+1]
'''
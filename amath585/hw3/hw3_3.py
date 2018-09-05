import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)

res = np.zeros((2,20))
for m in [10,20]:
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
    res[int(m/10-1),0:m] = u
    print(h,np.sqrt(h)*np.linalg.norm(u-(1-x[1:])**2))

#<startTeX>
# richardson extrapolation
m = 10
h = 1/m
x = np.linspace(0,1,m+1)
richardson = (4*res[1,1::2] - res[0,0:10])/3

plt.scatter(x[1:],((1-x[1:])**2)) #actual solution
plt.scatter(x[1:],richardson) # numerical solution
print("m=10:",np.sqrt(h)*np.linalg.norm(res[0,0:10]-(1-x[1:])**2))
print("m=20 (subsampled):",np.sqrt(h)*np.linalg.norm(res[1,1::2]-(1-x[1:])**2))
print("richardson:",np.sqrt(h)*np.linalg.norm(richardson-(1-x[1:])**2))
#<endTeX>
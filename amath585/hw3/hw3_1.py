import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=16)

#<startTeX>
# problem discritization
def G(theta):
    Gout = np.zeros(m+1)
    for i in range(1,m):
        Gout[i] = (theta[i-1]-2*theta[i]+theta[i+1])/h2+np.sin(theta[i])
    return Gout[1:m+1] # return only inner things since boundaries are fixed

def J(theta):
    Jout = np.triu(np.tril(np.ones((m,m)),1),-1)-np.identity(m) # trigiagonal all ones
    Jout += np.diag(-2 + h2*np.cos(theta[1:m+1])) 
    return Jout/h2

# problem parameters
alpha = 0.7
beta = 0.7
T = 2*np.pi # part a
x = np.linspace(0,T,m+2)
h = T/(m+1)
h2 = h**2

# discritization parameters
m = 512
x = np.linspace(0,T,m+2)  

#initial guess
theta = 0.7*np.cos(x)+0.5*np.sin(x)
for k in range(25):
        print(k)
        delta = np.linalg.solve(J(theta),-G(theta))
        theta[1:m+1] += delta
        
        if max(abs(delta)) < 10e-14:
            break

plt.figure()
plt.scatter(x,theta)
plt.savefig('img/1/original.pdf')

theta = 0.7 + abs(x-np.pi)-np.pi
for k in range(25):
        print(k)
        delta = np.linalg.solve(J(theta),-G(theta))
        theta[1:m+1] += delta
        
        if max(abs(delta)) < 10e-14:
            break

plt.figure()
plt.scatter(x,theta)
plt.savefig('img/1/abs.pdf')

maxtheta = []
theta = 0.7 + np.sin(x/2)
for T in np.linspace(6,62,8):

    x = np.linspace(0,T,m+2)
    h = T/(m+1)
    h2 = h**2

    # Newton's method
    for k in range(25):
        print(k)
        delta = np.linalg.solve(J(theta),-G(theta))
        theta[1:m+1] += delta
        
        if max(abs(delta)) < 10e-14:
            break

    maxtheta = np.append(maxtheta,max(abs(theta)))
    plt.figure()
    plt.scatter(x,theta)
    plt.savefig('img/1/'+str(int(T))+'.pdf')

print(maxtheta)
#<endTeX>
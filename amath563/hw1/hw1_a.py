#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#<start>
import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)

import matplotlib.pyplot as plt

import scipy as sp
from scipy.interpolate import interp1d
from scipy import integrate

from data_driven_modeling import bin_exp,diff,diff2,lasso, kl_divergence, AIC, BIC

#%% setup data

# manually input data
hare_population = np.array([20,20,52,83,64,68,83,12,36,150,110,60,7,10,70,100,92, 70,10,11,137,137,18,22,52,83,18,10,9,65])
lynx_population = np.array([32,50,12,10,13,36,15,12,6,6,65,70,40,9,20,34,45,40,15, 15,60,80,26,18,37,50,35,12,12,25])

# make population vector from data
P = np.array([hare_population,lynx_population])

# set up times data was taken (in years since first data point)
t = np.linspace(0,60,30,endpoint=False)


#%% interpolate data
P_interp = interp1d(t, P, kind='cubic',axis=1) 

# evaluate interpolation on a finer mesh
mesh_scale = 8
t_fine = np.linspace(0,58,30*mesh_scale-(mesh_scale-1))
P_ = P_interp(t_fine)
    

# plot interpolation vs. data
plt.plot(t_fine,P_[0],'.8');
plt.plot(t,P[0],color='0',marker='o',ms=5,linestyle='None')
plt.plot(t_fine,P_[1],'.8',linestyle='--');
plt.plot(t,P[1],color='0',marker='s',ms=5,linestyle='None')
plt.xlabel('time (years)')
plt.ylabel('population (thousands)')
plt.savefig('img/P/interp.pdf')

# plot interpolation phase plot
plt.figure()
plt.plot(P_[0],P_[1],color='.8');
plt.plot(P[0],P[1],color='0',marker='o',linestyle='None');
plt.xlabel('hare population')
plt.ylabel('lynx population')
plt.savefig('img/P/phase.pdf')


#%% dP/dt = kth degree polynomial in entries of P

def diff_model(t_fine, P_, max_deg, num_refinements):
    # compute time derivative of data
    dt = t_fine[1]-t_fine[0]
    dP = diff(P_,dt,endpoints=True,axis=1)
    

    # make dictionary of polynomials in both population1
    A = bin_exp(P_,max_deg) 
    weights  = np.sum(A,axis = 0)

    # compute coefficeints
    c = np.linalg.lstsq(A,dP.T)[0]
    # c[:,0] = lasso(A,dP[0].T,alpha=100)
    # c[:,1] = lasso(A,dP[1].T,alpha=100)
    
    params = np.ones(np.shape(c))

    # iteritively remove some functions from library
    for i in range(2):
        for j in range(num_refinements[i]):

            # find nonzero entries
            nz = np.nonzero(params[:,i])[0] 
            
            # find index of lowest weight of these entires
            x = np.argmin( weights[nz] * c[nz,i] ) 
            
            # zero this index
            params[nz[x],i] = 0 
                    
            # compute new coefficeints
            c[:,i] = np.linalg.lstsq(A*params[:,i],dP[i].T)[0]
    
    # solve solution from given coefficeints
    def rhs(t,xy):
        return bin_exp(xy,max_deg)@c
    ivp = integrate.solve_ivp(rhs,[0,60],P_[:,0],dense_output = True)
        
    return {"lib": lambda xy: bin_exp(xy,max_deg)@c, "model": ivp.sol, "num_params":np.sum(params,axis=0), "num_data": len(t_fine)}

#%% test various models

for max_deg, num_refinements in [[1,[0,0]],[2,[0,0]],[5,[0,0]],[1,[1,1]],[2,[1,1]],[2,[2,2]],[5,[1,1]]]:
    
    d_m = diff_model(t_fine,P_,max_deg,num_refinements)

    lib = d_m['lib']
    model = d_m['model']
    num_data = d_m['num_data']
    num_params = d_m['num_params']

    bins = [np.arange(-25,200+1,12.5),np.arange(-25,200+1,12.5)]
    eps = 1
    
    center= np.average(P,axis=1)
    
    # compute historgram of data and model in phase space
    p_ = np.histogram2d(P[0],P[1],[[-np.inf,center[0],np.inf],[-np.inf,center[1],np.inf]])[0]
    m_ = np.histogram2d(model(t)[0],model(t)[1],[[-np.inf,center[0],np.inf],[-np.inf,center[1],np.inf]])[0]
    
    p_ /= np.sum(p_)
    m_ /= np.sum(m_)
    
    # compute derivative of model at data points
    h = (t_fine[1]-t_fine[0])
    derivs = (model(t+h) - model(t-h)) / (2*h)

    # compute library times data points
    libs = lib(P).T
    
    # compute KL divergence
    data = np.array([kl_divergence(p_,m_)])
    np.savetxt('img/P/'+str(max_deg)+str(num_refinements[0])+str(num_refinements[1])+'.txt',data, fmt='%.4f')

    # compute AIC/BIC scores for each population
    for i in range(2):
        scores = {"AIC" : AIC(libs[i],derivs[i],num_params[i]),
                  "BIC": BIC(libs[i],derivs[i],num_params[i])}

        data = np.array([[scores['AIC'], scores['BIC']]])
        np.savetxt('img/P/'+str(max_deg)+str(num_refinements[0])+str(num_refinements[1])+'_'+str(i)+'.txt',data, fmt='%.3f & %.3f')

    
    # plot model vs actual data
    plt.figure()
    plt.plot(t_fine,P_[0],'.8');
    plt.plot(t_fine,P_[1],'.8',linestyle='--');
    plt.plot(t_fine,model(t_fine)[0],color='0',linestyle='-')
    plt.plot(t_fine,model(t_fine)[1],color='0',linestyle='--')
    plt.xlabel('time (years)')
    plt.ylabel('population (thousands)')
    plt.savefig('img/P/'+str(max_deg)+str(num_refinements[0])+str(num_refinements[1])+'.pdf')
    

#%% construct time delay matrices

# evaluate interpolation on a finer mesh
mesh_scale = 50
t_fine = np.linspace(0,58,30*mesh_scale-(mesh_scale-1))
P_ = P_interp(t_fine)

depth = 20
width = np.shape(P_)[1] - depth
H = np.zeros((2,depth,width))
H[0] = np.array([P_[0,i:width + i] for i in np.arange(depth)])
H[1] = np.array([P_[1,i:width + i] for i in np.arange(depth)])

# take SVD
u,s,vh = np.linalg.svd(H)

# plot singular values
plt.figure()
ax = plt.gca()
ax.set_yscale('log')
plt.plot(np.arange(depth),s[0]/np.max(s[0]),color='0',marker='o',ms=5,linestyle='None')
plt.plot(np.arange(depth),s[1]/np.max(s[1]),color='0',marker='s',ms=5,linestyle='None')
plt.savefig('img/P/time_delay.pdf')
#<end>


#%% 
'''
try kth degree linear difference equation
# never really finished here

n = len(P_[0])
k = 3
A = np.ones((n-k,2*(k+1)+1))
for i in np.arange(k+1):
    A[:,i] = P_[0,i:n-k+i]
    A[:,k+i+1] = P_[1,i:n-k+i]

ch = lasso(A[:,2:],A[:,0],10000)
cl = lasso(A[:,2:],A[:,1],10000)

#%% model: P_i = 5th degree polynomials of P_{i-1}

# evaluate interpolation on finer mesh
t_fine = np.linspace(0,58,30*3-2)
P_ = P_interp(t_fine)

max_deg = 5
A_fd5 = bin_exp(P_[:,:-1],max_deg)
c_fd5 = np.linalg.lstsq(A_fd5,P_[:,1:].T)[0]
#c_fd5 = lasso(A_fd5, P_[:,1:].T, alpha = 1e-4).T


# step using difference equation found above
max_iter = len(t_fine)
model_fd5 = np.zeros((2,max_iter))
model_fd5[:,0] = P_[:,0]

for k in range(max_iter-1):
    model_fd5[:,k+1] = bin_exp(model_fd5[:,k],max_deg)@c_fd5

# plot model vs actual data
plt.figure()
plt.plot(t_fine,P_[0],'.8');
plt.plot(t_fine,P_[1],'.9');
plt.plot(t_fine,model_fd5[0],t_fine,model_fd5[1])
plt.show()

#%%  plot coefficients of model

plt.figure();
plt.bar(np.arange(len(c_fd5[:,0])),c_fd5[:,0]);
plt.show();
plt.figure();
plt.bar(np.arange(len(c_fd5[:,1])),c_fd5[:,1]);
plt.show();

#%% test various models

for mesh_scale, max_deg, num_refinements in [[1,1,[0,0]],[2,2,[2,2]],[3,4,[6,9]],[9,5,[13,15]]]:
    
    # cv points
    cv_Q = True
    cv_points = 5*mesh_scale
    cv_start = None
    if cv_Q:
        cv_start = - cv_points
    
    t_fine = np.linspace(0,58,30*mesh_scale-(mesh_scale-1))
    P_ = P_interp(t_fine)
    
    d_m = diff_model(t_fine[:cv_start],P_[:,:cv_start],max_deg,num_refinements)

    model = d_m['model'](t_fine)
    num_data = d_m['num_data']
    num_params = d_m['num_params']
    print({"mesh_size": len(t_fine), "num_params": num_params})

    bins = [np.arange(-25,200+1,12.5),np.arange(-25,200+1,12.5)]
    eps = 1
    
    # compute KL divergence, AIC, BIC
    for i in range(2):
        p = np.histogram(P_[i,cv_start:],bins=bins[i])[0]
        p[p == 0] += eps
        p = p / np.sum(p)
        
        p_m = np.histogram(model[i,cv_start:],bins=bins[i])[0]
        p_m[p_m == 0] += eps
        p_m = p_m / np.sum(p_m)
                
        scores = {"kl_divergence" : kl_divergence(p,p_m),
                  "AIC" : AIC(model[i,cv_start:],P_[i,cv_start:],num_params[i]),
                  "BIC": BIC(model[i,cv_start:],P_[i,cv_start:],num_params[i])}

        data = np.array([[num_data, num_params[0], scores['kl_divergence'], scores['AIC'], scores['BIC']]])
        np.savetxt('img/'+str(mesh_scale)+str(max_deg)+str(num_refinements[0])+str(num_refinements[1])+'_'+str(i)+'.txt',data, fmt='%i & %i & %.3f & %.3f & %.3f')
    
    print(scores)
    
    # plot model vs actual data
    plt.figure()
    plt.plot(t_fine,P_[0],'.8');
    plt.plot(t_fine,P_[1],'.8',linestyle='--');
    plt.plot(t_fine,model[0],color='0',linestyle='-')
    plt.plot(t_fine,model[1],color='0',linestyle='--')
    if cv_Q:
        plt.axvspan(58+58*cv_start/num_data, 58, alpha=0.3, color='.8')
    plt.savefig('img/P/'+str(mesh_scale)+str(max_deg)+str(num_refinements[0])+str(num_refinements[1])+'.pdf')
    
'''
#%%




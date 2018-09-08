get_ipython().magic('pylab inline')

import numpy as np
import tensorflow as tf

from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras import optimizers
from keras.models import Model,Sequential,load_model
from keras.layers import Input,Dense, Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model

from IPython.display import clear_output


# for fun
def progress_bar(percent):
    length = 40
    pos = round(length*percent)
    clear_output(wait=True)
    print('['+'█'*pos+' '*(length-pos)+']  '+str(int(100*percent))+'%')


# define system
def lrz_rhs(t,x,sigma,beta,rho):
    return [sigma*(x[1]-x[0]), x[0]*(rho-x[2]), x[0]*x[1]-beta*x[2]];


# wrapper to generate trajecotry
end_time = 8
sample_rate = 100
t = np.linspace(0,end_time,sample_rate*end_time+1,endpoint=True)
def lrz_trajectory(rho):
    sigma=10;
    beta=8/3;
    x0 = 20*(np.random.rand(3)-.5)
    sol = integrate.solve_ivp(lambda t,x: lrz_rhs(t,x,sigma,beta,rho),[0,end_time],x0,t_eval=t,rtol=1e-10,atol=1e-11)
    return sol.y


# plot trajectory
x = lrz_trajectory(28)
plt.figure()
plt.gca(projection='3d')
plt.plot(x[0],x[1],x[2])
plt.show()


# ## Generate Data
N = 200
T = 801
rhos = [10,28,40]
input_data = np.zeros((N*(T-1)*len(rhos),4))
target_data = np.zeros((N*(T-1)*len(rhos),3))
for k,rho in enumerate(rhos):
    for i in range(N):
        progress_bar((N*k+i+1)/(N*len(rhos)))
        trajectory = lrz_trajectory(rho)
        input_data[((len(rhos)-1)*k+i)*(T-1):((len(rhos)-1)*k+i+1)*(T-1),:3] = trajectory.T[:-1]
        input_data[((len(rhos)-1)*k+i)*(T-1):((len(rhos)-1)*k+i+1)*(T-1),3] = rho
        target_data[((len(rhos)-1)*k+i)*(T-1):((len(rhos)-1)*k+i+1)*(T-1),:3] = trajectory.T[1:]


# ## Define Neural Network
# set up keras
def rad_bas(x):
    return K.exp(-x**2)
get_custom_objects().update({'rad_bas': Activation(rad_bas)})

def tan_sig(x):
    return 2/(1+K.exp(-2*x))-1
get_custom_objects().update({'tan_sig': Activation(tan_sig)})

# define neural net
model = Sequential()
model.add(Dense(10, activation='tan_sig', use_bias=True, input_shape=(4,)))
model.add(Dense(10, activation='sigmoid', use_bias=True))
model.add(Dense(10, activation='linear', use_bias=True))
model.add(Dense(3))

# set up loss function and optimizer
adam1 = optimizers.Adam(lr=.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=True, clipvalue=0.5)
model.compile(loss='mean_squared_error', optimizer=adam1, metrics=['accuracy'])

# train data
model.fit(input_data, target_data, epochs=10000, batch_size=1000, shuffle=True, callbacks=[plot_losses], validation_split=0.0)


# ## Test NN Predictions
rho=35
xsol = lrz_trajectory(rho)
x = np.zeros((3,end_time*sample_rate+1))
x[:,0] = xsol[:,0]
for i in range(end_time*sample_rate):
    x[:,i+1] = model.predict(np.array([np.append(x[:,i],rho)]))

# plot actual trajectory vs NN predicted trajectory
mpl.rcParams['text.usetex'] = True
plt.figure()
plt.gca(projection='3d')
plt.plot(x[0],x[1],x[2], color='k', linestyle=':')
plt.plot(xsol[0],xsol[1],xsol[2], color='k')
plt.savefig('img/NN_lrz35_trajectory.pdf')

for i in range(3):
    plt.figure()
    plt.plot(t,x[i], color='k')
    plt.plot(t,xsol[i], color='k', linestyle=':')
    plt.savefig('img/NN_lrz35_trajectory_'+str(i)+'.pdf')


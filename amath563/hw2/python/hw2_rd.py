get_ipython().magic('pylab inline')

import numpy as np
import tensorflow as tf

from scipy import integrate
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras import optimizers
from keras.models import Model,Sequential,load_model
from keras.layers import Input,Dense,Reshape,Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model

from IPython.display import clear_output


# ## set up keras
def rad_bas(x):
    return K.exp(-x**2)
get_custom_objects().update({'rad_bas': Activation(rad_bas)})

def tan_sig(x):
    return 2/(1+K.exp(-2*x))-1
get_custom_objects().update({'tan_sig': Activation(tan_sig)})


# ## Load RD trajectories
N = 128
T = 201
num_iter = 20
num_tests = 1
RD_all_data = np.zeros((num_iter-num_tests,T,N,2*N))
RD_input_data = np.zeros(((T-1)*(num_iter-num_tests),N,2*N))
RD_target_data = np.zeros(((T-1)*(num_iter-num_tests),N,2*N))

for i in range(num_iter-num_tests):
    d = loadmat('PDECODES/RD_data/N'+str(N)+'/iter'+str(i+1)+'.mat')
    u = d['u']
    v = d['v'] 
    RD_all_data[i,:,:,:N] = u[:,:,:].T
    RD_all_data[i,:,:,N:] = v[:,:,:].T
    RD_input_data[i*(T-1):(i+1)*(T-1),:,:] = RD_all_data[i,:-1,:,:]
    RD_target_data[i*(T-1):(i+1)*(T-1),:,:] = RD_all_data[i,1:,:,:]


RD_test_data = np.zeros((T*num_tests,N,2*N))
for i in range(num_tests):
    d = loadmat('PDECODES/RD_data/N'+str(N)+'/iter'+str(num_iter-i)+'.mat')
    u = d['u']
    v = d['v']
    RD_test_data[i*T:(i+1)*T,:,:N] = u.T
    RD_test_data[i*T:(i+1)*T,:,N:] = v.


# ## Train Neural Network

# define neural net
model = Sequential()
model.add(Dense(N*2*N, activation='tan_sig', use_bias=True, input_shape=(N*2*N,)))
model.add(Dense(N*2*N, activation='sigmoid', use_bias=True))
model.add(Dense(N*2*N))

# set up loss function and optimizer
adam1 = keras.optimizers.Adam(lr=.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=True, clipvalue=0.5)
model.compile(loss='mean_squared_error', optimizer=adam1, metrics=['accuracy'])

# train data
model.fit(
    np.reshape(RD_input_data,(-1,N*2*N)), 
    np.reshape(RD_target_data,(-1,N*2*N)), 
    epochs=1000, batch_size=800, shuffle=True, callbacks=[plot_losses], validation_split=0.0)

# ## Test Neural Network
# compute NN trajectory
RD_NN_prediction = np.zeros(np.reshape(RD_test_data[0:T],(-1,N*2*N)).shape)
RD_NN_prediction[0] = np.reshape(RD_test_data[0],(-1,N*2*N))
for k in range(T-1):
    RD_NN_prediction[k+1] = model.predict(np.array([RD_NN_prediction[k]]))

# plot NN trajectory
mpl.rcParams['text.usetex'] = True
i=80
m = plt.pcolormesh(np.reshape(RD_NN_prediction,(-1,N,2*N))[i])
m.set_rasterized(True)
plt.axis('image')
plt.savefig('img/predicted_RD_'+str(i)+'_trajectory.pdf')

# ## Compute SVD
# Reshape data and compute rank k approximation to find fixed subspace to which we project our spatial ponints at each time.

RD_all_data_reshaped = np.reshape(RD_all_data[:,:,:,],(-1,2*N*N)).T
[uu,ss,vvh] = np.linalg.svd(RD_all_data_reshaped,full_matrices=False)

# plot singular values
mpl.rcParams['text.usetex'] = True
plt.scatter(np.arange(len(ss)),ss,color='k')
plt.savefig('img/singular_values.pdf')

# plot SVD modes
i=0
plt.figure(figsize=(6,3.3))
m = plt.pcolormesh(np.reshape(uu[:,i],(N,2*N)))
m.set_rasterized(True)
plt.axis('image')
plt.savefig('img/svd_mode_'+str(i+1)+'.pdf')
plt.gcf().get_size_inches()

plt.figure(figsize=(6,3.3))
plt.plot(np.arange(len(vvh[i])),vvh[i],color='k')
plt.savefig('img/svd_coeff_'+str(i+1)+'.pdf')
plt.gcf().get_size_inches()


# set rank and take reduced SVD
rank = 100
u = uu[:,:rank]
s = ss[:rank]
vh = vvh[:rank]

# set up trainign data for new NN
SVD_input_data = np.delete(vh,np.s_[200::201],axis=1).T
SVD_target_data = np.delete(vh,np.s_[1::201],axis=1).T
SVD_input_data.shape

# plot rank reduced image in time vs actual image in time
plt.figure(figsize=(6,3.3))
m = plt.pcolormesh(np.reshape(u@np.diag(s)@SVD_input_data[180],(N,2*N)))
m.set_rasterized(True)
plt.axis('image')
plt.savefig('img/uv_t180.pdf')

plt.figure(figsize=(6,3.3))
m = plt.pcolormesh(RD_all_data[0,180])
m.set_rasterized(True)
plt.axis('image')
plt.savefig('img/svd_t180.pdf')


# ## Train Net on SVD data
# define neural net
model = Sequential()
model.add(Dense(2*rank, activation='tan_sig', use_bias=True, input_shape=(rank,)))
model.add(Dense(2*rank, activation='sigmoid', use_bias=True))
model.add(Dense(2*rank, activation='linear', use_bias=True))
model.add(Dense(rank))

# set up loss function and optimizer
adam1 = keras.optimizers.Adam(lr=.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=True, clipvalue=0.5)
model.compile(loss='mean_squared_error', optimizer=adam1, metrics=['accuracy'])

# train data
model.fit(
    SVD_input_data, 
    SVD_target_data,
    epochs=1000, batch_size=80, shuffle=True, callbacks=[plot_losses], validation_split=0.0)

# set up data for testing
SVD_test_data = np.reshape(RD_test_data[0:T],(-1,N*2*N))@u

# compute NN trajectory
SVD_NN_prediction = np.zeros(SVD_test_data.shape)
SVD_NN_prediction[0] = SVD_test_data[0]
for k in range(T-1):
    SVD_NN_prediction[k+1] = model.predict(np.array([SVD_NN_prediction[k]]))


# plot NN trajectory at given time
plt.figure(figsize=(6,3.3))
m = plt.pcolormesh(np.reshape(u@np.diag(s)@SVD_NN_prediction[15],(N,2*N)))
m.set_rasterized(True)
plt.axis('image')
plt.savefig('img/svd_prediction_t15.pdf')

# plot NN prediction vs actual trajectory after SVD
plt.figure()
plt.scatter(np.arange(T),SVD_NN_prediction[:,0])
plt.scatter(np.arange(T-1),SVD_input_data[:,0])
plt.show()

# animate NN trajectory
get_ipython().magic('matplotlib notebook')
import matplotlib.animation

t = np.arange(T)
fig, ax = plt.subplots()

def animate(i):
    plt.pcolormesh(np.reshape(u@np.diag(s)@SVD_NN_prediction[i],(N,2*N)))
    plt.axis('image')

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(t))
plt.show()




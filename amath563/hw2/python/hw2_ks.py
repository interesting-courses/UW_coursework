get_ipython().magic('pylab inline')

import numpy as np
import tensorflow as tf

from scipy import integrate
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras import optimizers
from keras.models import Model,Sequential,load_model
from keras.layers import Input,Dense, Activation
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

# ## Load KS trajectories
N = 1024
T = 251
num_iter = 40
num_tests = 1
KS_input_data = np.zeros(((T-1)*num_iter,N))
KS_target_data = np.zeros(((T-1)*num_iter,N))

for i in range(num_iter-num_tests):
    u = loadmat('PDECODES/KS_data/N'+str(N)+'/iter'+str(i+1)+'.mat')['uu']
    KS_input_data[i*(T-1):(i+1)*(T-1)] = u[:,:-1].T
    KS_target_data[i*(T-1):(i+1)*(T-1)] = u[:,1:].T

# save data to test on outside of training data
KS_test_data = np.zeros((T*num_tests,N))
for i in range(num_tests):
    u = loadmat('PDECODES/KS_data/N'+str(N)+'/iter'+str(num_iter-i)+'.mat')['uu']
    KS_test_data[i*T:(i+1)*T] = u.T

# plot test trajectory
mpl.rcParams['text.usetex'] = True
m = plt.pcolormesh(KS_test_data)
m.set_rasterized(True)
plt.savefig('img/sample_KS_trajectory.pdf')


# ## Train Neural Network
# define neural net
model = Sequential()
model.add(Dense(2*N, activation='tan_sig', use_bias=True, input_shape=(N,)))
model.add(Dense(N))

# set up loss function and optimizer
adam1 = keras.optimizers.Adam(lr=.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=True, clipvalue=0.5)
model.compile(loss='mean_squared_error', optimizer=adam1, metrics=['accuracy'])

# train data
model.fit(KS_input_data, KS_target_data, epochs=1000, batch_size=3000, shuffle=True, validation_split=0.0)

# ## Test Neural Network
# compute NN trajectory
KS_NN_prediction = np.zeros(KS_test_data[0:T].shape)
KS_NN_prediction[0] = KS_test_data[0]
for k in range(T-1):
    KS_NN_prediction[k+1] = model.predict(np.array([KS_NN_prediction[k]]))

# plot NN trajectory
mpl.rcParams['text.usetex'] = True
m = plt.pcolormesh(KS_NN_prediction)
m.set_rasterized(True)
plt.savefig('img/predicted_KS_trajectory.pdf')

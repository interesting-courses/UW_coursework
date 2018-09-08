get_ipython().magic('pylab inline')

import numpy as np
import tensorflow as tf

from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D

import keras
from keras import optimizers
from keras.models import Model,Sequential,load_model
from keras.layers import Input,Dense,Reshape,Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model

from IPython.display import clear_output


# define system
def lrz_rhs(t,x,sigma,beta,rho):
    return [sigma*(x[1]-x[0]), x[0]*(rho-x[2]), x[0]*x[1]-beta*x[2]];

# wrapper to genrate trajectory
end_time = 10
sample_rate = 100
t = np.linspace(0,end_time,sample_rate*end_time+1,endpoint=True)
def lrz_trajectory(rho):
    sigma=10;
    beta=8/3;
    x0 = 20*(np.random.rand(3)-.5)
    sol = integrate.solve_ivp(lambda t,x: lrz_rhs(t,x,sigma,beta,rho),[0,end_time],x0,t_eval=t,rtol=1e-10,atol=1e-11)
    return sol.y


# ## Categorize data by lobe
# Pick seperating hyperplane $0 = c^Tx$ where $c=(5,1,0)$

mpl.rcParams['text.usetex'] = True

# separate data to left and right nodes
c=([5,1,0])
L=x[:,np.where((c@x).T>=0)[0]]
R=x[:,np.where((c@x).T<0)[0]]

# plot left and right nodes of trajectory in 3d space
plt3d = plt.figure().gca(projection='3d')
plt3d.scatter(L[0],L[1],L[2],marker='.')
plt3d.scatter(R[0],R[1],R[2],marker='.')
plt.show()


# ## Generate Data
# label data with how far a given point is to crossing
def generate_timed_trajectory(rho):
    # define normal vector
    c = np.array([5,1,0])
    
    # get trajectory
    x = lrz_trajectory(rho)
    
    # classify points as left or right
    classes = np.sign(c@x)
    # compute which indicies correspond to transitions by 
    #    checking if there is a sign change in classification of points
    transition_ind=np.where(np.convolve(classes,[1,-1],mode='valid')!=0)
    
    # this will be the time it takes to the next jump
    # start with ones where there are jumps
    time_to_jump = np.zeros((len(x.T)))
    time_to_jump[transition_ind] = np.ones(len(transition_ind))    

    # count backwards from ones
    current_time_to_jump = 0
    jumping = False
    for j in range(len(time_to_jump)):
        if time_to_jump[-j]==1:
            current_time_to_jump=1
            jumping=True
        elif jumping:
            time_to_jump[-j]=current_time_to_jump
            current_time_to_jump+=1
    
    # delete end of data where we do not know how long until the next crossing
    ind_to_save = np.where(time_to_jump!=0)[0]
    clipped_data = np.vstack([x,time_to_jump])[:,ind_to_save]
    
    return clipped_data

# generate data over multiple trajectories
max_iter = 100
D = np.empty((4,0))
for i in range(max_iter):
    D = np.concatenate([D,generate_timed_trajectory(28)],axis=1)

# format data for training
input_data = D[:3].T
target_data = D[3].T


# ## Set up Neural Network
# set up keras
def rad_bas(x):
    return K.exp(-x**2)
get_custom_objects().update({'rad_bas': Activation(rad_bas)})

def tan_sig(x):
    return 2/(1+K.exp(-2*x))-1
get_custom_objects().update({'tan_sig': Activation(tan_sig)})


# define neural net
model = Sequential()
model.add(Dense(10, activation='tan_sig', use_bias=True, input_shape=(3,)))
model.add(Dense(10, activation='sigmoid', use_bias=True))
model.add(Dense(10, activation='linear', use_bias=True))
model.add(Dense(1))

# set up loss function and optimizer
adam1 = optimizers.Adam(lr=.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=True, clipvalue=0.5)
model.compile(loss='mean_squared_error', optimizer=adam1, metrics=['accuracy'])

# train data
model.fit(input_data, target_data, epochs=1000, batch_size=1000, shuffle=True, callbacks=[plot_losses], validation_split=0.0)


# ## Test NN Predictions
# see how model predicts over an entire trajectory.
x = generate_timed_trajectory(28)

next_time_to_jump = np.zeros(len(x.T))
for k,xpos in enumerate(x.T):
    next_time_to_jump[k] = model.predict(np.array([xpos[:3]]))


mpl.rcParams['text.usetex'] = True
# plot trajectory and separating hyperplane
plt.figure()
plt.plot(np.linspace(-7,5),-5*np.linspace(-7,5),color='.6',linestyle='--')
plt.plot(x[0],x[1],color='k')
plt.xlabel('$x$-axis')
plt.ylabel('$y$-axis')
plt.savefig('img/separating_hyerplane.pdf')

# plot actual time to crossing and NN predicted time to crossing
plt.figure()
plt.plot(np.arange(len(x.T)),x[3], color='k')
plt.plot(np.arange(len(x.T)),next_time_to_jump, color='k', linestyle=':')
plt.ylabel('iteractions until jump')
plt.savefig('img/jump_predictor.pdf')




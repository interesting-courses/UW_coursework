#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
import numpy as np
from scipy.io import netcdf
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation
import datetime

# import netCDF files and set up as numpy arrays
sst_nc = netcdf.netcdf_file('sst.wkmean.1990-present.nc', 'r')
mask_nc = netcdf.netcdf_file('lsmask.nc', 'r')

time = sst_nc.variables['time'][:].copy()
sst = np.transpose(sst_nc.variables['sst'][:].copy(),axes=[1,2,0])
mask = mask_nc.variables['mask'][:].copy()[0,:,:]

sst_nc.close()
mask_nc.close()

N = 360*180;
L = 1400;

M,N,T = np.shape(sst)
max_temp = np.max(sst)
min_temp = np.min(sst)

# apply mask to all entries
sst = sst*np.transpose([mask]*T,axes=[1,2,0])

%matplotlib inline

#%% convert times since 1800 to dates
def time_to_date(t):
    origin_time = datetime.date(1800,1,1)
    delta = datetime.timedelta(days=t)
    return origin_time + delta

dates = list(map(time_to_date,time))

#%% take SVD of 3d array
#<start:svd_slice>
def svd_of_slice(s):
    sst_cut = sst[s]
    M,N,T = np.shape(sst_cut)

    sst_reshaped = np.reshape(sst_cut,(-1,T))
    [u,s,vh] = np.linalg.svd(sst_reshaped, full_matrices=False)

    U = np.reshape(u, (M,N,-1))

    return U,s,vh
#<end:svd_slice>

#%% load ONI data from NOAA
    
ONI = np.array([0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.3, 0.4, 0.4, #1990
0.4, 0.3, 0.2, 0.3, 0.5, 0.6, 0.7, 0.6, 0.6, 0.8, 1.2, 1.5,
1.7, 1.6, 1.5, 1.3, 1.1, 0.7, 0.4, 0.1, -0.1, -0.2, -0.3, -0.1,
0.1, 0.3, 0.5, 0.7, 0.7, 0.6, 0.3, 0.3, 0.2, 0.1, 0.0, 0.1,
0.1, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.6, 0.7, 1.0, 1.1,
1.0, 0.7, 0.5, 0.3, 0.1, 0.0, -0.2, -0.5, -0.8, -1.0, -1.0, -1.0,
-0.9, -0.8, -0.6, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.4, -0.4, -0.5,
-0.5, -0.4, -0.1, 0.3, 0.8, 1.2, 1.6, 1.9, 2.1, 2.3, 2.4, 2.4,
2.2, 1.9, 1.4, 1.0, 0.5, -0.1, -0.8, -1.1, -1.3, -1.4, -1.5, -1.6,
-1.5, -1.3, -1.1, -1.0, -1.0, -1.0, -1.1, -1.1, -1.2, -1.3, -1.5, -1.7,
-1.7, -1.4, -1.1, -0.8, -0.7, -0.6, -0.6, -0.5, -0.5, -0.6, -0.7, -0.7,
-0.7, -0.5, -0.4, -0.3, -0.3, -0.1, -0.1, -0.1, -0.2, -0.3, -0.3, -0.3,
-0.1, 0.0, 0.1, 0.2, 0.4, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.1,
0.9, 0.6, 0.4, 0.0, -0.3, -0.2, 0.1, 0.2, 0.3, 0.3, 0.4, 0.4,
0.4, 0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7,
0.6, 0.6, 0.4, 0.4, 0.3, 0.1, -0.1, -0.1, -0.1, -0.3, -0.6, -0.8,
-0.8, -0.7, -0.5, -0.3, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9,
0.7, 0.3, 0.0, -0.2, -0.3, -0.4, -0.5, -0.8, -1.1, -1.4, -1.5, -1.6,
-1.6, -1.4, -1.2, -0.9, -0.8, -0.5, -0.4, -0.3, -0.3, -0.4, -0.6, -0.7,
-0.8, -0.7, -0.5, -0.2, 0.1, 0.4, 0.5, 0.5, 0.7, 1.0, 1.3, 1.6,
1.5, 1.3, 0.9, 0.4, -0.1, -0.6, -1.0, -1.4, -1.6, -1.7, -1.7, -1.6,
-1.4, -1.1, -0.8, -0.6, -0.5, -0.4, -0.5, -0.7, -0.9, -1.1, -1.1, -1.0,
-0.8, -0.6, -0.5, -0.4, -0.2, 0.1, 0.3, 0.3, 0.3, 0.2, 0.0, -0.2,
-0.4, -0.3, -0.2, -0.2, -0.3, -0.3, -0.4, -0.4, -0.3, -0.2, -0.2, -0.3,
-0.4, -0.4, -0.2, 0.1, 0.3, 0.2, 0.1, 0.0, 0.2, 0.4, 0.6, 0.7,
0.6, 0.6, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.5, 2.6,
2.5, 2.2, 1.7, 1.0, 0.5, 0.0, -0.3, -0.6, -0.7, -0.7, -0.7, -0.6])

ONI_dates = []
for y in range(1990,2017):
    for m in range(1,13):
        ONI_dates.append(datetime.date(y,m,1))
        
ONI_interp_f = sp.interpolate.interp1d(np.linspace(0,1,len(ONI)),ONI)
ONI_interp = ONI_interp_f(np.linspace(0,1,len(dates)))

#%% Take SVD of full data
        
U,s,vh = svd_of_slice(np.s_[:,:,:])


#%% Plot singular values
mpl.rcParams['text.usetex'] = True
plt.figure()
plt.scatter(np.arange(16),s[:16],color='k')
plt.yscale('log')
plt.savefig('img/svd/'+str(T)+'/s.pdf',bbox_inches='tight')

#%% Plot SVD modes

for j in range(8):
    plt.figure(figsize=(6,3))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # plot ONI data
    ax2.fill_between(ONI_dates,0,ONI,color='.8')
    
    # plot filtered data
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    ax1.plot(dates,s[j]*vh[j],color='k')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))

    plt.savefig('img/svd/'+str(T)+'/vh_'+str(j)+'.pdf',bbox_inches='tight')
    
    
    plt.figure(figsize=(6,3))
    
    ax1 = plt.gca()
    
    ax1.invert_yaxis()
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])

    m = plt.imshow(U[:,:,j] , vmin=-5e-3,vmax=9e-3)
    m.set_rasterized(True)
#    plt.axis('image')
    
    plt.savefig('img/svd/'+str(T)+'/u_'+str(j)+'.pdf',bbox_inches='tight')

#%% Try part of 5th SVD mode
        
X = np.copy(U[:,:,5])

for i in range(180):
    for j in range(360):
        if np.abs(i-89)**3/20**3+np.abs(j-230)**3/30**3 >=1:
            X[i,j] = 0
            
X = sp.ndimage.filters.gaussian_filter(X,3)

coeffs = np.zeros(T)
for t in range(T):
    coeffs[t] = np.sum(X*sst[:,:,t])

plt.figure(figsize=(6,3))
plt.gca().invert_yaxis()
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])

m = plt.imshow(X)
m.set_rasterized(True)
#    plt.axis('image')

plt.savefig('img/el_nino_mode.pdf',bbox_inches='tight')


#%% Try single single point instead
coeffs = sst[90,215]

#%% Plot raw projection

plt.figure(figsize=(6,3))
ax = plt.gca()
plt.ylabel('temperature (degree C)')
plt.xlabel('date')

ax2 = ax.twinx()
ax2.fill_between(ONI_dates,0,ONI,color='.8')
ax.plot(dates,coeffs/100,color='k')
plt.ylabel('ONI')

ax.set_zorder(ax2.get_zorder()+1)
ax.patch.set_visible(False)
#ax.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
plt.savefig('img/el_nino_coeff.pdf',bbox_inches='tight')

#%% Plot Filtered projection

# define low pass filter
b, a = signal.butter(4, .036)

plt.figure(figsize=(6,3))
ax1 = plt.gca()
plt.ylabel('temperature (degree C)')

ax2 = ax1.twinx()

# plot ONI data
ax2.fill_between(ONI_dates,0,ONI,color='.8')
plt.ylabel('ONI')

# plot filtered data
ax1.set_zorder(ax2.get_zorder()+1)
ax1.patch.set_visible(False)
#ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
ax1.plot(dates,coeffs/100,color='.5')
ax1.plot(dates,sp.signal.filtfilt(b, a, coeffs/100, method="gust"),color='k')

plt.savefig('img/el_nino_filtered.pdf',bbox_inches='tight')

#%% Take DWT

widths = np.arange(1,100)
cwtmatr = signal.cwt(coeffs, signal.ricker, widths)
m = plt.pcolormesh(dates,widths,cwtmatr)
m.set_rasterized(True)
plt.savefig('img/el_nino_wavelet_transform.pdf',bbox_inches='tight')


#%% Try Time Slice

U,s,vh = svd_of_slice(np.s_[:,:,361:413])

plt.figure()
plt.scatter(np.arange(16),s[:16],color='k')
plt.yscale('log')
plt.show()
#plt.savefig('img/svd/'+str(T)+'/s.pdf',bbox_inches='tight')

for j in range(8):
    plt.figure(figsize=(6,3))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # plot ONI data
#    ax2.fill_between(ONI_dates,0,ONI,color='.8')
    
    # plot filtered data
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    ax1.plot(dates[361:413],s[j]*vh[j],color='k')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))

    plt.savefig('img/svd/shortT/vh_'+str(j)+'.pdf',bbox_inches='tight')
    
    
    plt.figure(figsize=(6,3))
    
    ax1 = plt.gca()
    
    ax1.invert_yaxis()
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])

    m = plt.imshow(U[:,:,j] , vmin=-5e-3,vmax=9e-3)
    m.set_rasterized(True)
    plt.axis('image')
    plt.savefig('img/svd/shortT/u_'+str(j)+'.pdf',bbox_inches='tight')

#%% Try Spatial Slice

U,s,vh = svd_of_slice(np.s_[70:110,150:300,:])

plt.figure()
plt.scatter(np.arange(16),s[:16],color='k')
plt.yscale('log')
plt.show()
#plt.savefig('img/svd/'+str(T)+'/s.pdf',bbox_inches='tight')

for j in range(8):
    plt.figure(figsize=(6,3))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # plot ONI data
    ax2.fill_between(ONI_dates,0,ONI,color='.8')
    
    # plot filtered data
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    ax1.plot(dates,s[j]*vh[j],color='k')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))

    plt.savefig('img/svd/smallMN/vh_'+str(j)+'.pdf',bbox_inches='tight')
    
    
    plt.figure(figsize=(6,2))
    
    ax1 = plt.gca()
    
    ax1.invert_yaxis()
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])

    m = plt.imshow(U[:,:,j] , vmin=-5e-3,vmax=9e-3)
    m.set_rasterized(True)
    plt.axis('image')
    plt.savefig('img/svd/smallMN/u_'+str(j)+'.pdf',bbox_inches='tight')

#%% Set up Data for Neural Net

ONI_threshold = 2;

# classify points as above or below threshold
classes = np.sign(ONI_interp-ONI_threshold)

# get points where ONI first goes above ONI_threshold
transition_ind=np.where(np.convolve(classes,[1,-1],mode='valid')>0)

time_to_jump = np.zeros(T)
time_to_jump[transition_ind] = np.ones(len(transition_ind))    

# count backwards from ones
current_time_to_jump = 0
jumping = False
for j in range(T):
    if time_to_jump[-j]==1:
        current_time_to_jump=1
        jumping=True
    elif jumping:
        time_to_jump[-j]=current_time_to_jump
        current_time_to_jump+=1

last_known_jump_index = np.where(time_to_jump!=0)[0][-1]

#%% Train Neural Network

import keras
from keras import optimizers
from keras.models import Model,Sequential,load_model
from keras.layers import Input,Dense,Reshape,Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model

def rad_bas(x):
    return K.exp(-x**2)
get_custom_objects().update({'rad_bas': Activation(rad_bas)})

def tan_sig(x):
    return 2/(1+K.exp(-2*x))-1
get_custom_objects().update({'tan_sig': Activation(tan_sig)})

input_data = np.reshape(sst,(-1,T))[:,:,last_known_jump_index]
target_data = time_to_jump[:last_known_jump_index]

#%%
model = Sequential()
model.add(Dense(M*N, activation='tan_sig', use_bias=True, input_shape=(M*N,)))
model.add(Dense(M*N, activation='sigmoid', use_bias=True))
model.add(Dense(M*N, activation='linear', use_bias=True))
model.add(Dense(1))


adam1 = optimizers.Adam(lr=.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=True, clipvalue=0.5)
model.compile(loss='mean_squared_error', optimizer=adam1, metrics=['accuracy'])


model.fit(input_data, target_data, epochs=1000, batch_size=1000, shuffle=True, validation_split=0.05)

#%%
def progress_bar(percent):
    length = 40
    pos = round(length*percent)
#    clear_output(wait=True)
    return('['+'█'*pos+' '*(length-pos)+']  '+str(int(100*percent))+'%')

#%% Animate Full Data

%matplotlib auto

fig, ax = plt.subplots(figsize=(5, 3))
fig.gca().invert_yaxis()
#ax.axis('image')

mesh = ax.pcolormesh(sst[:,:,0]*mask,vmin=min_temp,vmax=max_temp)
fig.colorbar(mesh)

def animate(i):
    mesh.set_array((sst[:,:,i]).reshape(-1))
    plt.title(progress_bar(i/T),name='ubuntu mono')

    
anim = FuncAnimation(
    fig, animate, interval=6, frames=T-1)
 
plt.draw()
plt.show()

# parameters of the system
import os
import numpy as np
import scipy
from scipy import optimize

import copy
import pdb
import sys
import time
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM,BatchNormalization, Concatenate
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D

from keras.callbacks import EarlyStopping

import theano
import theano.tensor as T
import keras.backend as K
import tensorflow as tf

# disable warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

import gc; gc.collect()


#cwd = os.getcwd()
#print(cwd)





#import data files
dir_data = './forecast_lib/data/'
temp = np.load(dir_data + 'temp_norm.npy')
total_load = np.load(dir_data + 'total_load_norm.npy')
load_meters = np.load(dir_data + 'load_meters_norm.npy')


# system parameters
T, num_meters = load_meters.shape
num_samples = T


####################################################
# training parameters
hist_samples = 24

train_samples = int(0.8 * num_samples)
test_samples_a = int(0.1 * num_samples)

# define the time interval for learning and evaluating the attack 
t_0_train = train_samples
t_f_train = train_samples + test_samples_a

t_0_test = t_f_train
t_f_test = T


y = total_load
y_test = y[t_0_test+hist_samples:t_f_test]


use_mse = False

if use_mse:
	loss_f = 'mean_squared_error'
	loss = 'mse'
else:
	loss_f = 'mean_absolute_error'
	loss = 'mae'

early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=False, min_delta=0.001)



# optimizer
optim = 'L-BFGS-B'

from .forecast_training import train_and_save_forecaster, extract_data
from .forecast_attack import find_attack, MAE, get_forecast



import numpy as np
import os 
import time
import random

import forecast_lib as fl
import forecast_lib.forecast_training as ft
import forecast_lib.forecast_attack as fa


dropout=False

if dropout:
	type_exp = '_dropout'
	rho_d = -1
else:
	type_exp = ''
	rho_d = 1

# experiment parameters
directory = './experiments/MAE_vs_MSE'+type_exp+'/'

m = fl.num_meters

max_num_models = 10

m_d = m
#m_d = int(0.5*m)

type_loss = ['mean_squared_error', 'mean_absolute_error']
loss_short = ['mse', 'mae']

for i in range(len(type_loss)):

	dir_rep = directory + 'loss_' + loss_short[i] + '/'
	try:
		os.makedirs(dir_rep)
	except:
		pass

	ft.loss_f = type_loss[i]

	t0 = time.perf_counter()
	print('loss: ' + loss_short[i])

	for k in range(max_num_models):
		list_meters = []
		rand_set = random.sample( set(range( m )), m_d )
		list_meters.append( rand_set )

		np.save(dir_rep + 'meters_' + str(k) + '.npy', list_meters)

		data = fl.extract_data( list_meters )

		model_name = 'model_' + str(k) + '_0'
		fl.train_and_save_forecaster(dir_rep, data, model_name, rho_d)

	t_f = time.perf_counter()
	print('\t***Train time: ' + str((t_f-t0)/60.0))





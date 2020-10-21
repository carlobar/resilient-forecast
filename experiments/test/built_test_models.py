import numpy as np
import os 
import time
import random
import pdb

import forecast_lib as fl
import forecast_lib.forecast_training as ft

dropout=False

if dropout:
	type_exp = '_dropout'
	rho_d = -1
else:
	type_exp = ''
	rho_d = 1

# experiment parameters
directory = './experiments/test'+type_exp+'/'

m = fl.num_meters

max_num_models = 10



n_vec = [1, 2, 2, 2]

for i in range(len(n_vec)):
	print('model = '+str(i))
	dir_rep = directory + 'model_'+str(i) +'/' 
	try:
		os.makedirs(dir_rep)
	except:
		pass

	#m_d = m_d_exp[i]
	n = n_vec[i]


	t0 = time.perf_counter()
	print('\t n: '+str(n))

	for k in range(max_num_models):
		model_name = 'model_' + str(k)

		list_meters = ft.get_partition( m, n )

		np.save(dir_rep + 'meters_' + str(k) + '.npy', list_meters)

		#data = fl.extract_data( list_meters )

		# define the target for the experiment
		
		if i == 0:
			ft.y = fl.total_load
			data = fl.extract_data( list_meters )
			fl.train_and_save_forecaster(dir_rep, data, model_name, rho_d)
			#pdb.set_trace()
		elif i==1:
			ft.y = fl.total_load
			data = fl.extract_data( list_meters )
			fl.train_and_save_forecaster(dir_rep, data, model_name, rho_d)

		elif i == 2:
			ft.y = fl.total_load * 0.5
			data = fl.extract_data( list_meters )
			fl.train_and_save_forecaster(dir_rep, data, model_name, rho_d)
		
		if i in [3]:
			for j in range(n):
				# get the total load for the meters
				total_load = np.sum(fl.load_meters, axis=1)
				max_load = max(total_load)
				ft.y = np.sum(fl.load_meters[:, list_meters[j]], axis=1)/max_load
				# extract the data for this set
				model_name_j = model_name + '_' + str(j)
				data_j = fl.extract_data( [list_meters[j]] )
				fl.train_and_save_forecaster(dir_rep, data_j, model_name_j, rho_d)



	t_f = time.perf_counter()
	print('\t***Train time: ' + str((t_f-t0)/60.0))





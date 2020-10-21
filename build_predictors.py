
import numpy as np
import os 
import time
import random

import forecast_lib as fl


dropout=False

if dropout:
	type_exp = '_dropout'
	rho_d = -1
else:
	type_exp = ''
	rho_d = 1

# experiment parameters
directory = './experiments/models_diff_size'+type_exp+'/'

m = fl.num_meters

max_num_models = 20

#m_d_frac = np.linspace(0.1, 0.4, 4)
m_d_frac = np.linspace(0.5, 1, 5)

for i in range(len(m_d_frac)):

	m_d = int(m * m_d_frac[i])
	dir_rep = directory + 'm_d_' + str(m_d) + '/'
	try:
		os.makedirs(dir_rep)
	except:
		pass

	t0 = time.perf_counter()
	print('\t m_d: '+str(m_d))

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





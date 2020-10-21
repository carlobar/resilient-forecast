import numpy as np
import os 
import time
import random

import forecast_lib as fl


dropout=False

if dropout:
	type_exp = '_dropout'
else:
	type_exp = ''

# experiment parameters
directory = './experiments/models_diff_size'+type_exp+'/'

m = fl.num_meters

max_num_models = 20

m_d_frac = np.linspace(0.5, 1, 5)

m_a_frac = np.linspace(0.1, 0.5, 5)

reps = 20

unique_bias = True


strategic_attack=False

if strategic_attack:
	type_exp='strategic_' + type_exp
else:
	type_exp=''+ type_exp


impact = np.zeros((reps, len(m_d_frac), len(m_a_frac)))
pred_error = np.zeros((reps, len(m_d_frac), len(m_a_frac)))

for i in range(len(m_d_frac)):

	m_d = int(m * m_d_frac[i])
	print('m_d: '+str(m_d))

	dir_models = directory + 'm_d_' + str(m_d) + '/'
	try:
		os.makedirs(dir_rep)
	except:
		pass

	for j in range(len(m_a_frac)):
		m_a = int(m_a_frac[j]*m)
		print('\tm_a='+str(m_a))

		t0 = time.perf_counter()


		for k in range(max_num_models):
			print('\t\tk='+str(k))

			if strategic_attack:
				meters_model = np.load(dir_models + 'meters_' + str(k) + '.npy',  allow_pickle=True)
				meters_a = random.sample( set( meters_model[0] ), m_a )
			else:
				meters_a = random.sample( set(range( m )), m_a )

			y_test, hat_y, hat_y_a, bias_opt = fl.find_attack(dir_models, max_num_models, 1, meters_a, unique_bias)

			impact[k, i, j] = fl.MAE(hat_y, hat_y_a)	
			pred_error[k, i, j] = fl.MAE(hat_y, y_test)	

		t_f = time.perf_counter()
		print('\t***Train time: ' + str((t_f-t0)/60.0))



dir_results = './'
np.save( dir_results + 'impact'+type_exp+'.npy', impact)
np.save( dir_results + 'pred_error'+type_exp+'.npy', pred_error)






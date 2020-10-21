import numpy as np
import os 
import time
import random

import matplotlib.pyplot as plt
from matplotlib import mlab

import forecast_lib as fl


max_num_models = 20
m = 109
m_d_frac = np.concatenate((np.linspace(0.1, 0.4, 4), np.linspace(0.5, 1, 5)))



dropout=True

if dropout:
	type_exp = '_dropout'
else:
	type_exp = ''


directory = './experiments/models_diff_size'+type_exp+'/'

# test the models
error = np.zeros((len(m_d_frac), max_num_models))
all_predictions = []

for i in range(len(m_d_frac)):
	m_d = int(m * m_d_frac[i])
	print('m_d='+str(m_d))
	dir_rep = directory + 'm_d_' + str(m_d) + '/'

	predictions_i = []

	t0 = time.perf_counter()
	print('\t m_d: '+str(m_d))

	for k in range(max_num_models):
		print('\tk='+str(k))
		y_test, hat_y = fl.get_forecast(dir_rep, k)
		predictions_i.append(hat_y)
		error[i, k] = fl.MAE(y_test, hat_y)	

	all_predictions.append(predictions_i)

	t_f = time.perf_counter()
	print('\t***Train time: ' + str((t_f-t0)/60.0))




dir_results = './'
np.save( dir_results + 'MAE'+type_exp+'.npy', error)
np.save( dir_results + 'predictions'+type_exp+'.npy', all_predictions)




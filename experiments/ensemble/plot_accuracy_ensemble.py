import numpy as np
import os 
import time
import random

import matplotlib.pyplot as plt
from matplotlib import mlab


import forecast_lib as fl

dropout=False

if dropout:
	type_exp = '_dropout'
else:
	type_exp = ''



strategic_attack=False

if strategic_attack:
	type_exp='strategic_' + type_exp
else:
	type_exp=''+ type_exp


type_exp += '_ensemble'

dir_results = './results/'

error = np.load(dir_results + 'MAE'+type_exp+'.npy',  allow_pickle=True)
predictions = np.load(dir_results + 'predictions'+type_exp+'.npy',  allow_pickle=True)

m = 109
models_len, num_models = error.shape

m_d_frac = np.array([0.2, 0.33, 0.5, 0.66, 0.8])
n_vec = [5, 3, 2, 3, 5]


avg_error = np.mean(error, axis=1)
'''
avg_error = np.zeros(5)
for i in range(models_len):
	if i in [0, 1]:
		c = n_vec[i] - 1
	else:
		c=1
	c=1
	avg_error[i] = np.mean(error[i, :])*c
'''

plt.figure(4)
plt.clf()
plt.plot(m_d_frac*m, avg_error, '--', label='Ensemble')
plt.plot(m_d_frac*m, np.ones(5)*0.057, label='Original model')
#for i in range(models_len):
#	plt.plot(error[i, :], label='exp '+str(i))
plt.title('Accuracy of the Ensemble')
plt.xlabel('Number of meters per model ($m^d$)')
plt.ylabel('Prediction error MAE($y, \hat y$)')
plt.legend()
plt.show()

plt.savefig('accuracy_ensemble.pgf', bbox_inches='tight')




#for i in range(models_len):
#	print('MAE exp ' + str(i) + '= ' + str(np.mean(error[i, :])))



k = 9
plt.figure(5)
plt.clf()
for i in range(models_len):
	'''	
	if i in [0, 1]:
		c = n_vec[i] - 1
	else:
		c=1
	'''
	c = 1
	pred_i = predictions[i][k].reshape(-1)*c
	mae_i = fl.MAE(fl.y_test, pred_i)


	plt.plot(pred_i, label='exp '+str(i)+', MAE='+str(mae_i))
plt.plot(fl.y_test, '--', label='Test')
plt.legend()
plt.show()








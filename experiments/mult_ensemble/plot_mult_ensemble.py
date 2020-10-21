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


type_exp += '_mult_ensemble'

dir_results = './results/'

impact = np.load(dir_results + 'impact'+type_exp+'.npy',  allow_pickle=True)
pred_error = np.load(dir_results + 'pred_error'+type_exp+'.npy',  allow_pickle=True)

m = 109
reps, max_num_models = impact.shape





plt.figure(1)
plt.clf()
plt.plot(range(1, max_num_models+1), np.mean(impact, axis=0))
plt.ylim((0, 1))
plt.xlabel('Number of ensembles')
plt.ylabel('Impact ($\hat y - \hat y_a$)')
plt.title('Impact with Multiple Ensembles')
plt.show()

plt.savefig('impact_mult_ensemble.pgf', bbox_inches='tight')




plt.figure(2)
plt.clf()
plt.plot(range(1, max_num_models+1), np.mean(pred_error, axis=0))
plt.ylim((0, 1))
plt.xlabel('Number of ensembles')
plt.ylabel('Prediction error MAE($y, \hat y_a$)')
plt.title('Prediction Error with Multiple Ensembles')
plt.show()

plt.savefig('pred_error_mult_ensemble.pgf', bbox_inches='tight')

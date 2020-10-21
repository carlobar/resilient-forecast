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

#type_exp += '_test'
type_exp += '_test'

dir_results = './results/'

error = np.load(dir_results + 'MAE'+type_exp+'.npy',  allow_pickle=True)
predictions = np.load(dir_results + 'predictions'+type_exp+'.npy',  allow_pickle=True)

m = 109
models_len, num_models = error.shape


plt.figure(4)
plt.clf()
for i in range(models_len):
	plt.plot(error[i, :], label='exp '+str(i))
plt.title('Accuracy')
plt.legend()
plt.show()

for i in range(models_len):
	print('MAE exp ' + str(i) + '= ' + str(np.mean(error[i, :])))



k = 9
plt.figure(5)
plt.clf()
for i in range(models_len):
	pred_i = predictions[i][k].reshape(-1)
	mae_i = fl.MAE(fl.y_test, pred_i)
	plt.plot(pred_i, label='exp '+str(i)+', MAE='+str(mae_i))
plt.plot(fl.y_test, '--', label='Test')
plt.legend()
plt.show()








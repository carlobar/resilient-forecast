import numpy as np
import os 
import time
import random

import matplotlib.pyplot as plt
from matplotlib import mlab




dropout=False

if dropout:
	type_exp = '_dropout'
else:
	type_exp = ''


error = np.load('MAE'+type_exp+'.npy',  allow_pickle=True)
predictions = np.load('predictions'+type_exp+'.npy',  allow_pickle=True)

m = 109
m_d_len, num_models = error.shape
m_d_frac = np.concatenate((np.linspace(0.1, 0.4, 4), np.linspace(0.5, 1, 5)))

avg_error = np.mean(error, axis=1)
std_error = np.std(error, axis=1)


plt.figure(1)
plt.clf()
plt.plot(m_d_frac*m, avg_error)
plt.title('Prediction Error')
plt.xlabel('$m_d$')
plt.ylabel('MAE')
plt.legend()
plt.show()



plt.figure(2)
plt.clf()
for j in range(num_models):
	plt.plot(m_d_frac*m, error[:, j])
plt.show()



'''
for i in range(len(predictions)):
	plt.figure(3+i)
	plt.clf()

	predictions_i = predictions[i]
	for j in range(len(predictions_i)):
		plt.plot(predictions_i[j])
	plt.show()
'''

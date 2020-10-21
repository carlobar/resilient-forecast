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



strategic_attack=False

if strategic_attack:
	type_exp='strategic_' + type_exp
else:
	type_exp=''+ type_exp


impact = np.load('impact'+type_exp+'.npy',  allow_pickle=True)
pred_error = np.load('pred_error'+type_exp+'.npy',  allow_pickle=True)

m = 109
reps, len_m_d, len_m_a = impact.shape

m_d_frac = np.linspace(0.5, 1, len_m_d)
m_a_frac = np.linspace(0.1, .5, len_m_a)



plt.figure(1)
plt.clf()
for i in range(len_m_d):
	m_d = int(m_d_frac[i]*m)
	impact_i = impact[:, i, :]
	avg_impact_i = np.mean(impact_i, axis=0)
	plt.plot(m_a_frac*m, avg_impact_i, label='$m_d=$'+str(m_d))
plt.legend()
plt.ylim((0, 1))
plt.show()


i=0
plt.figure(2)
plt.clf()
m_d = int(m_d_frac[i]*m)
impact_i = impact[:, i, :]
for k in range(reps):
	plt.plot(m_a_frac*m, impact_i[k, :])
plt.legend()
plt.title('$m_d=$'+str(m_d))
plt.ylim((0, 1))
plt.show()


plt.figure(3)
plt.clf()
for i in range(len_m_d):
	m_d = int(m_d_frac[i]*m)
	pred_error_i = pred_error[:, i, :]
	avg_pred_error_i = np.mean(pred_error_i, axis=0)
	plt.plot(m_a_frac*m, avg_pred_error_i, label='$m_d=$'+str(m_d))
plt.legend()
plt.ylim((0, 1))
plt.show()


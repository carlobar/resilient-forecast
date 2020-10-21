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



type_exp += '_test'

dir_results = './results/'

impact = np.load(dir_results + 'impact'+type_exp+'.npy',  allow_pickle=True)[0:10, :, :]
pred_error = np.load(dir_results + 'pred_error'+type_exp+'.npy',  allow_pickle=True)[0:10, :, :]

impact_b = np.load(dir_results + 'impact'+type_exp+'_b.npy',  allow_pickle=True)[0:10, :, :]
pred_error_b = np.load(dir_results + 'pred_error'+type_exp+'_b.npy',  allow_pickle=True)[0:10, :, :]

impact = np.load(dir_results + 'impact'+type_exp+'_c.npy',  allow_pickle=True)
pred_error = np.load(dir_results + 'pred_error'+type_exp+'_c.npy',  allow_pickle=True)



m = 109
reps, len_models, len_m_a = impact.shape

#n_vec = [1, 2, 2, 2]

m_a_frac = np.linspace(0.01, .5, len_m_a)


label=['$F^1$ ($m^d = m$, $y_i=y$)', '$F^2$ ($m^d = 0.5 m$, $y_i=y$)', '$F^3$ ($m^d = 0.5 m$, $y_i = 0.5 y$)', '$F^4$ ($m^d = 0.5 m$, $y_i = \sum_{i\in M^d} l_i$)']

line=['','--',':','-.']

plt.figure(1)
plt.clf()
for i in range(len_models):
	impact_i = impact[:, i, :]
	avg_impact_i = np.mean(impact_i, axis=0)
	plt.plot(m_a_frac*m, avg_impact_i, line[i],  label=label[i])
plt.legend()
plt.ylim((0, 0.5))
plt.title('Impact of Attacks on Different Models')
plt.xlabel('Number of Attacks ($m_a$)')
plt.ylabel('Impact ($\hat y - \hat y_a$)')
plt.show()


plt.savefig('impact_models.pgf', bbox_inches='tight')





i=3
plt.figure(6)
plt.clf()
impact_i = impact[:, i, :]
for k in range(reps):
	plt.plot(m_a_frac*m, impact_i[k, :])
plt.legend()
plt.title('$exp =$'+str(i))
plt.ylim((0, 1))
plt.show()




plt.figure(3)
plt.clf()
for i in range(len_models):
	pred_error_i = pred_error[:, i, :]
	avg_pred_error_i = np.mean(pred_error_i, axis=0)
	plt.plot(m_a_frac*m, avg_pred_error_i, label='$exp=$'+str(i))
plt.legend()
plt.ylim((0, 1))
plt.title('Accuracy')
plt.show()

'''
i=0
plt.figure(7)
plt.clf()
for k in range(reps):
	pred_error_i = pred_error[k, i, :]
	plt.plot(m_a_frac*m, pred_error_i, label='$exp=$'+str(i))

plt.ylim((0, 1))
plt.title('Accuracy exp = '+str(i))
plt.show()
'''




plt.figure(10)
plt.clf()
impact_nom = np.mean( impact[:, 0, :],  axis=0)
plt.plot(m_a_frac*m, impact_nom, '--', label='$exp nom$')

impact_05 = np.mean( impact[:, 3, :],  axis=0)
plt.plot(m_a_frac*m, impact_05, label='$exp 0.5$')
for i in range(len_models):
	impact_i = impact_b[:, i, :]
	avg_impact_i = np.mean(impact_i, axis=0)
	plt.plot(m_a_frac*m, avg_impact_i, label='$exp =$'+str(i))
plt.legend()
plt.ylim((0, 1))
plt.title('impact')
plt.show()


plt.figure(10)
plt.clf()
impact_nom = np.mean( impact[:, 0, :],  axis=0)
plt.plot(m_a_frac*m, impact_nom, '--', label='$exp nom$')

impact_05 = np.mean( impact[:, 3, :],  axis=0)
plt.plot(m_a_frac*m, impact_05, label='$exp 0.5$')
for i in range(len_models):
	impact_i = impact_b[:, i, :]
	avg_impact_i = np.mean(impact_i, axis=0)
	plt.plot(m_a_frac*m, avg_impact_i, label='$exp =$'+str(i))
plt.legend()
plt.ylim((0, 1))
plt.title('impact')
plt.show()


impact_new = np.concatenate((impact_b[:,0:2, :], impact[:, 3:4, :], impact_b[:,2:4, :]), axis=1)
plt.figure(11)
plt.clf()
for j in range(5):
	impact_j = np.mean( impact_new[:, :, j], axis=0 )
	plt.plot(impact_j, label='m_a='+str(j))

plt.legend()
plt.show()





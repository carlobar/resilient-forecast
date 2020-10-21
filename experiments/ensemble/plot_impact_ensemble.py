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




type_exp_norm = '_nominal'

dir_results = './results/'

impact_normal = np.load(dir_results + 'impact'+type_exp_norm+'.npy',  allow_pickle=True)[0:10, 0, :]

avg_impact_normal = np.mean(impact_normal, axis=0)
#m_a_frac_b = np.linspace(0.1, .5, 5)




type_exp += '_ensemble'


impact_a = np.load(dir_results + 'impact'+type_exp+'_a.npy',  allow_pickle=True)
pred_error_a = np.load(dir_results + 'pred_error'+type_exp+'_a.npy',  allow_pickle=True)

impact_b = np.load(dir_results + 'impact'+type_exp+'_b.npy',  allow_pickle=True)
pred_error_b = np.load(dir_results + 'pred_error'+type_exp+'_b.npy',  allow_pickle=True)

impact = np.concatenate((impact_a, impact_b), axis=0)


m = 109
reps, len_models, len_m_a = impact.shape



m_d_frac = np.array([0.2, 0.33, 0.5, 0.66, 0.8])
n_vec = [5, 3, 2, 3, 5]
m_a_frac = np.linspace(0.01, 0.5, 6)


#label=['$F_1$ ($m^d = m$, $y_i=y$)', '$F_2$ ($m^d = 0.5 m$, $y_i=y$)', '$F_3$ ($m^d = 0.5 m$, $y_i = 0.5 y$)', '$F_4$ ($m^d = 0.5 m$, $y_i = \sum_{i\in M^d} l_i$)']


linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'


linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]




linestyles=[1, 4, 8, 11, 2]

plt.figure(1)
plt.clf()
for i in range(len_models):

	if i in [0, 1]:
		c = n_vec[i] - 1
	else:
		c=1

	m_d = int(m_d_frac[i]*m)
	impact_i = impact[:, i, :]
	avg_impact_i = np.mean(impact_i, axis=0) * c

	(name, style) = linestyle_tuple[ linestyles[i] ]

	plt.plot(m_a_frac*m, avg_impact_i, linestyle=style,  label='$m^d$='+str(m_d_frac[i])+'m')

plt.plot(m_a_frac*m, avg_impact_normal, label='$m^d=m$')

plt.legend()
plt.ylim((0, 0.4))
plt.title('Impact of Attacks on the Ensemble')
plt.xlabel('Number of Attacks ($m_a$)')
plt.ylabel('Impact ($\hat y - \hat y_a$)')
plt.show()


plt.savefig('impact_ensemble.pgf', bbox_inches='tight')



'''
i=0
plt.figure(6)
plt.clf()
impact_i = impact[:, i, :]
for k in range(reps):
	plt.plot(m_a_frac*m, impact_i[k, :])
plt.legend()
plt.title('$exp =$'+str(i))
plt.ylim((0, 1))
plt.show()
'''

'''

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
'''

plt.figure(11)
plt.clf()
for j in range(len(m_a_frac)-1):

	(name, style) = linestyle_tuple[ linestyles[j] ]
	impact_j = np.mean( impact[:, :, j], axis=0 )
	plt.plot(m_d_frac*m, impact_j, linestyle=style, label='m_a='+str(round(m_a_frac[j], 2) )+'m')
j+=1
impact_j = np.mean( impact[:, :, j], axis=0 )
plt.plot(m_d_frac*m, impact_j, '-', label='m_a='+str(round(m_a_frac[j], 2) )+'m')

plt.title('Impact of Attacks on the Ensemble')
plt.xlabel('Meters used in each model ($m_d$)')
plt.ylabel('Impact ($\hat y - \hat y_a$)')
plt.legend()
plt.show()


plt.savefig('impact_ensemble_m_d.pgf', bbox_inches='tight')

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




#type_exp = '_mae_vs_mse'
#type_exp = '_mae_vs_mse_b'
type_exp = '_mae_vs_mse_not_unique'

dir_results = './results/'

impact = np.load(dir_results + 'impact'+type_exp+'.npy',  allow_pickle=True)



m = 109
reps, len_loss, len_m_a = impact.shape

m_a_frac = np.linspace(0.01, .9, 10)

type_loss = ['mean_squared_error', 'mean_absolute_error']
loss_short = ['mse', 'mae']


avg_impact_mse = np.mean(impact[:, 0, :], axis=0)
avg_impact_mae = np.mean(impact[:, 1, :], axis=0)



plt.figure(1)
plt.clf()
plt.plot(m_a_frac, avg_impact_mse, label='MSE')
plt.plot(m_a_frac, avg_impact_mae, label='MAE')
plt.legend()
plt.show()


plt.savefig('impact_mae_vs_mse.pgf', bbox_inches='tight')

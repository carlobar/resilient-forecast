import numpy as np
import os 
import time
import random

import forecast_lib as fl
import forecast_lib.forecast_training as ft
import forecast_lib.forecast_attack as fa


# function that aggregates the predictions of the individual models
def get_prediction_a(x, models):
	# get the prediction from each model
	predictions = fa.f_vec(x, models) 
	n = len(predictions)

	aggregate_predictions = sum(predictions) / float(n)

	return aggregate_predictions


def get_grad_a(x, list_meters, grad_f):
	grad_models = fa.grad_vec(x, list_meters, grad_f)
	n = len(grad_models)
	avg_grad = sum(grad_models) / float(n)
	return avg_grad



# function that aggregates the predictions of the individual models
def get_prediction_b(x, models):
	# get the prediction from each model
	predictions = fa.f_vec(x, models) 
	n = len(predictions)

	aggregate_predictions = sum(predictions) 

	return aggregate_predictions


def get_grad_b(x, list_meters, grad_f):
	grad_models = fa.grad_vec(x, list_meters, grad_f)
	n = len(grad_models)
	avg_grad = sum(grad_models)
	return avg_grad





dropout=False

if dropout:
	type_exp = '_dropout'
else:
	type_exp = ''

# experiment parameters
directory = './experiments/MAE_vs_MSE'+type_exp+'/'

m = fl.num_meters

max_num_models = 10

m_a_frac = np.linspace(0.01, .9, 10)

reps = 40

unique_bias = False


strategic_attack=False

if strategic_attack:
	type_exp='strategic_' + type_exp
else:
	type_exp=''+ type_exp


type_loss = ['mean_squared_error', 'mean_absolute_error']
loss_short = ['mse', 'mae']



impact = np.zeros((reps, len(type_loss), len(m_a_frac)))
pred_error = np.zeros((reps, len(type_loss), len(m_a_frac)))

for i in range(len(type_loss)):
	print('loss = '+type_loss[i])

	dir_models =  directory + 'loss_'+ loss_short[i] +'/' 

	for j in range(len(m_a_frac)):
		m_a = int(m_a_frac[j]*m)
		print('\tm_a='+str(m_a))

		t0 = time.perf_counter()


		for k in range(reps):
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



type_exp += '_mae_vs_mse_not_unique'

dir_results = './results/'
np.save( dir_results + 'impact'+type_exp+'.npy', impact)
np.save( dir_results + 'pred_error'+type_exp+'.npy', pred_error)






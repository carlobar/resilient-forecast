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



# function that aggregates the predictions of the individual models
def get_prediction_c(x, models):
	# get the prediction from each model
	predictions = fa.f_vec(x, models) 
	n = len(predictions)

	aggregate_predictions = sum(predictions) / (n-1)

	return aggregate_predictions


def get_grad_c(x, list_meters, grad_f):
	grad_models = fa.grad_vec(x, list_meters, grad_f)
	n = len(grad_models)
	avg_grad = sum(grad_models) / (n-1)
	return avg_grad






dropout=False

if dropout:
	type_exp = '_dropout'
else:
	type_exp = ''

# experiment parameters
directory = './experiments/ensemble'+type_exp+'/'


m = fl.num_meters

max_num_models = 10

m_a_frac = np.linspace(0.01, 0.5, 6)

reps = 20

unique_bias = True


strategic_attack=False

if strategic_attack:
	type_exp='strategic_' + type_exp
else:
	type_exp=''+ type_exp



m_d_frac = [0.2, 0.33, 0.5, 0.66, 0.8]
n_vec = [5, 3, 2, 3, 5]


impact = np.zeros((reps, len(n_vec), len(m_a_frac)))
pred_error = np.zeros((reps, len(n_vec), len(m_a_frac)))

for i in range(len(n_vec)):
	print('model = '+str(i))

	dir_models =  directory + 'model_'+str(i) +'/' 

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



			if m_d_frac[i] >0.5:

				fa.get_prediction = get_prediction_c
				fa.get_grad = get_grad_c

			else:
				fa.get_prediction = get_prediction_b
				fa.get_grad = get_grad_b
			


			y_test, hat_y, hat_y_a, bias_opt = fl.find_attack(dir_models, max_num_models, 1, meters_a, unique_bias)

			impact[k, i, j] = fl.MAE(hat_y, hat_y_a)	
			pred_error[k, i, j] = fl.MAE(hat_y, y_test)	

		t_f = time.perf_counter()
		print('\t***Train time: ' + str((t_f-t0)/60.0))



type_exp += '_ensemble_b'

dir_results = './results/'
np.save( dir_results + 'impact'+type_exp+'.npy', impact)
np.save( dir_results + 'pred_error'+type_exp+'.npy', pred_error)






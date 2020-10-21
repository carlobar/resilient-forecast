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






# function that aggregates the predictions of the individual models
def get_prediction_d(x, models):
	# get the prediction from each model
	predictions = fa.f_vec(x, models) 
	n = len(predictions)

	aggregate_predictions = sum(predictions) / (n-num_models)

	return aggregate_predictions


def get_grad_d(x, list_meters, grad_f):
	grad_models = fa.grad_vec(x, list_meters, grad_f)
	n = len(grad_models)
	avg_grad = sum(grad_models) / (n-num_models)
	return avg_grad




# function that aggregates the predictions of the individual models
def get_prediction_e(x, models):
	# get the prediction from each model
	predictions = fa.f_vec(x, models) 
	n = len(predictions)

	aggregate_predictions = sum(predictions) / (num_models)

	return aggregate_predictions


def get_grad_e(x, list_meters, grad_f):
	grad_models = fa.grad_vec(x, list_meters, grad_f)
	n = len(grad_models)
	avg_grad = sum(grad_models) / (num_models)
	return avg_grad




dropout=False

if dropout:
	type_exp = '_dropout'
else:
	type_exp = ''

# experiment parameters
directory = './experiments/ensemble'+type_exp+'/'


m = fl.num_meters

max_num_models = 20

m_a_frac = np.linspace(0.01, 0.5, 6)

reps = 20

unique_bias = True


strategic_attack=False

if strategic_attack:
	type_exp='strategic_' + type_exp
else:
	type_exp=''+ type_exp



m_d_frac = [0.5]

models_vec = [1, 2, 3, 4, 5]

n_vec = [5, 3, 2, 3, 5]



dir_models =  directory + 'model_'+str(2) +'/' 

m_a = int(0.5*m)


impact = np.zeros((reps, len(models_vec)))
pred_error = np.zeros((reps, len(models_vec)))

for i in range(len(models_vec)):
	print('n = '+str(i))



	num_models = models_vec[i]

	t0 = time.perf_counter()


	for k in range(reps):
		print('\t\tk='+str(k))

		if strategic_attack:
			meters_model = np.load(dir_models + 'meters_' + str(k) + '.npy',  allow_pickle=True)
			meters_a = random.sample( set( meters_model[0] ), m_a )
		else:
			meters_a = random.sample( set(range( m )), m_a )



		if m_d_frac[0] >0.5:

			fa.get_prediction = get_prediction_d
			fa.get_grad = get_grad_d

		else:
			fa.get_prediction = get_prediction_e
			fa.get_grad = get_grad_e
			


		y_test, hat_y, hat_y_a, bias_opt = fl.find_attack(dir_models, max_num_models, num_models, meters_a, unique_bias)

		impact[k, i] = fl.MAE(hat_y, hat_y_a)	
		pred_error[k, i] = fl.MAE(hat_y, y_test)	

	t_f = time.perf_counter()
	print('\t***Train time: ' + str((t_f-t0)/60.0))



type_exp += '_mult_ensemble'

dir_results = './results/'
np.save( dir_results + 'impact'+type_exp+'.npy', impact)
np.save( dir_results + 'pred_error'+type_exp+'.npy', pred_error)






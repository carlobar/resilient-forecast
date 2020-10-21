import numpy as np
import os 
import time
import random
import pdb

import matplotlib.pyplot as plt
from matplotlib import mlab

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







max_num_models = 10
m = 109

n_vec = [3, 4, 3, 4]

dropout=False

if dropout:
	type_exp = '_dropout'
else:
	type_exp = ''


directory = './experiments/test_partitions'+type_exp+'/'

# test the models
error = np.zeros((len(n_vec), max_num_models))
all_predictions = []

for i in range(len(n_vec)):
	
	dir_rep = directory + 'model_' + str(i) + '/'

	predictions_i = []

	t0 = time.perf_counter()
	print('\t model: '+str(i))

	for k in range(max_num_models):
		print('\tk='+str(k))

		if i in [0, 1]:
			fa.get_prediction = get_prediction_b
			fa.get_grad = get_grad_b
	

		elif i in [2, 3]:
			fa.get_prediction = get_prediction_c
			fa.get_grad = get_grad_c
			
		y_test, hat_y = fa.get_forecast(dir_rep, k)
		predictions_i.append(hat_y)
		error[i, k] = fl.MAE(y_test, hat_y)	
		#pdb.set_trace()

	all_predictions.append(predictions_i)

	t_f = time.perf_counter()
	print('\t***Train time: ' + str((t_f-t0)/60.0))


type_exp += '_test_b'

dir_results = './results/'
np.save( dir_results + 'MAE'+type_exp+'.npy', error)
np.save( dir_results + 'predictions'+type_exp+'.npy', all_predictions)




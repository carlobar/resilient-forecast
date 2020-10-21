# functions used to attack the predictors

from forecast_lib import *

# The bias is periodic, this function returns the bias observed at time t
def adjust_bias(bias, t):
	H, m = bias.shape
	bias_b = np.zeros((H, m))
	for h in range(24):
		bias_b[h, :] = bias[(h-t)%24, :]
	return bias_b


# function that unpadets the historical data based on the bias
def update_data(list_meters, t_0, t_f,  bias, list_meters_bias):
	loads = load_meters[t_0:t_f, :]
	T, num_meters = loads.shape
	num_meters_a = len(list_meters_bias)
	num_samples = t_f - t_0 - hist_samples

	# construct the bias vector
	bias_ext = np.zeros((T, num_meters_a))
	for k in range(int(T / hist_samples)):
		bias_ext[k*hist_samples : (k+1)*hist_samples, :] = bias
	k += 1
	bias_ext[k*hist_samples : , :] = bias[0 : T - k*hist_samples, :]

	# apply the bias
	load_bias = copy.deepcopy( loads )
	for j in range(len(list_meters_bias)):
		meter = list_meters_bias[j]
		load_bias[:, meter] += bias_ext[:, j]
	temp_period = temp[t_0:t_f]

	num_sets = len(list_meters)

	input_data = []
	for k in range(num_sets):
		meters_k = list_meters[k]
		raw_data_k = load_bias[:, meters_k]
		data_k = np.zeros((num_samples, hist_samples, len(meters_k)+1))
		for t in range(num_samples):
			loads_k = raw_data_k[t : t + hist_samples, :]
			temp_k = temp_period[t : t + hist_samples]
			data_k[t, :, :] = np.concatenate((loads_k, temp_k.reshape(-1,1)), axis=1)
		input_data.append( data_k )

	return input_data


# prediction of the ith model with an input x
def f_i (x, models, i):
	return models[i].predict( x[i] )

# prediction of the models with an input x x
def f_vec(x, models):
	T, hist_samples, num_meters = x[0].shape
	n = len(models)
	pred = []
	for i in range(n):
		pred.append( f_i(x, models, i) )
	return pred


# function that aggregates the predictions of the individual models
def get_prediction(x, models):
	# get the prediction from each model
	predictions = f_vec(x, models) 
	n = len(predictions)

	aggregate_predictions = sum(predictions) / float(n)

	return aggregate_predictions


def grad_vec(x, list_meters, grad_f):
	T, H, m = x[0].shape
	n = len(x)

	avg_grad_i = np.zeros((H, num_meters))

	grad_models = []
	for i in range(n):

		meters_i = list_meters[i]
		x_i = x[i]
		grad_i = grad_f[i]( [x_i] )

		avg_grad_i = np.zeros((H, num_meters))
		for t in range(T):
			grad_t = grad_i[0][t]
			adjusted_bias = adjust_bias(grad_t, t)
			try:
				avg_grad_i[:, meters_i] += adjusted_bias[:, :-1]
			except:
				pdb.set_trace()

		grad_models.append( avg_grad_i/T )
	return grad_models


def get_grad(x, list_meters, grad_f):
	grad_models = grad_vec(x, list_meters, grad_f)
	n = len(grad_models)
	avg_grad = sum(grad_models) / n
	return avg_grad




def get_bias_matrix(bias_vec, meters_a):
	m_a = len(meters_a)
	n = len(bias_vec) / hist_samples
	# Check if we use a unique bias
	if n < m_a:
		bias = np.zeros((hist_samples, m_a))
		bias_base = bias_vec.reshape((hist_samples, 1))
	
		for i in range(m_a):
			bias[:, i] = bias_base[:, 0]
	else:	
		bias = bias_vec.reshape(hist_samples, m_a)
	return bias


def get_grad_vector(grad, meters_a, len_bias_vec):
	m_a = len(meters_a)
	n = len_bias_vec / hist_samples
	# Check if we use a unique bias
	if n < m_a:
		grad_vec = np.mean(grad[:, meters_a], axis=1).reshape(-1)
	else:	
		grad_vec = grad[:, meters_a].reshape(-1)
	return grad_vec


# Adversary's objective function that defines the attack's success as a function of the bias
def Gamma(bias_vec, models, grad_f, list_meters, t_0, t_f, meters_a):
	m_a = len(meters_a)
	bias = get_bias_matrix(bias_vec, meters_a)

	x_a = update_data(list_meters, t_0, t_f,  bias, meters_a)

	y_sample = y[t_0+hist_samples:t_f]
	try:
		hat_y_a = get_prediction(x_a, models)
	except:
		pdb.set_trace()
	error = hat_y_a - y_sample
	'''
	if np.any(error == np.nan ):
		pdb.set_trace()
	'''
	return np.mean(error)



# gradient of the objective function 
def grad_Gamma(bias_vec, models, grad_f, list_meters, t_0, t_f, meters_a):
	m_a = len(meters_a)
	bias = get_bias_matrix(bias_vec, meters_a)

	x_a = update_data(list_meters, t_0, t_f,  bias, meters_a)

	grad = get_grad(x_a, list_meters, grad_f)
	grad_a = get_grad_vector(grad, meters_a, len(bias_vec))
	'''
	if  np.any(grad_a == np.nan ):
		pdb.set_trace()
	'''
	return grad_a





def find_attack(directory, total_models, number_models, meters_a, unique_bias=False):
	#t_0 = time.perf_counter()

	# create a new session
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	K.set_session(sess)

	models = []
	grad_f = []

	m_a = len(meters_a)

	# select randomly the models
	id_models = sorted( random.sample( range(total_models), number_models ) )

	# Get the meters from the models
	list_meters = []
	for i in range(number_models):
		id_model = id_models[i]

		# load the data and the model of the forecastes
		meters_model = np.load(directory + 'meters_' + str(id_model) + '.npy',  allow_pickle=True)
		for j in range(len(meters_model)):
			list_meters.append( meters_model[j] )		

			if len(meters_model)>1:
				model_name = directory + 'model_' + str(id_model) + '_' + str(j) + '.h5'
			else:
				model_name = directory + 'model_' + str(id_model) + '.h5'

			try:
				model = load_model( model_name )
			except:
				try:
					model_name = directory + 'model_' + str(id_model) + '_' + str(j) + '.h5'
					model = load_model( model_name )
				except:
					pdb.set_trace()
			models.append( model )
			grad_f.append( K.function( [model.input], K.gradients( model.output, [model.input] ) ) )

	# construnct the initial condition
	if unique_bias:
		bias0 = np.zeros((hist_samples, 1))
		bias_rand = np.random.rand(hist_samples, 1)
	else:
		bias0 = np.zeros((hist_samples, m_a))
		bias_rand = np.random.rand(hist_samples, m_a)

	bias0_vec = bias0.reshape(-1)
	bias0_matrix = get_bias_matrix(bias0_vec, meters_a)

	bias_rand_vec = bias_rand.reshape(-1)
	bias_rand_matrix = get_bias_matrix(bias_rand_vec, meters_a)


	# find optimal attack
	#t_0 = time.perf_counter()

	solution = optimize.minimize(Gamma, bias_rand_vec, args=(models, grad_f, list_meters, t_0_train, t_f_train, meters_a), jac=grad_Gamma, method=optim)
	bias_opt_vec = solution.x
	bias_opt = get_bias_matrix(bias_opt_vec, meters_a)

	# calculate the impact on the test data
	x_0 = update_data(list_meters, t_0_test, t_f_test,  bias0, meters_a)
	x_a = update_data(list_meters, t_0_test, t_f_test,  bias_opt, meters_a)

	hat_y = get_prediction(x_0, models)
	hat_y_a = get_prediction(x_a, models)

	#y_test_b, hat_y_b = get_forecast(directory, 0)

	#pdb.set_trace()

	# clear the current session
	K.clear_session()
	del model, models

	return y_test, hat_y, hat_y_a, bias_opt



def get_forecast(directory, id_model):
	# create a new session
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	K.set_session(sess)

	models = []
	grad_f = []

	# Get the meters from the model
	list_meters = []

	# load the data and the model of the forecastes
	meters_model = np.load(directory + 'meters_' + str(id_model) + '.npy',  allow_pickle=True)
	for j in range(len(meters_model)):
		list_meters.append( meters_model[j] )
		if len(meters_model)>1:
			model_name = directory + 'model_' + str(id_model) + '_' + str(j) + '.h5'
		else:
			model_name = directory + 'model_' + str(id_model) + '.h5'
		#print('\tmodel: '+model_name)
		try:
			model = load_model( model_name )
		except:
			try:
				model_name = directory + 'model_' + str(id_model) + '_' + str(j) + '.h5'
				model = load_model( model_name )
			except:
				pdb.set_trace()
		models.append( model )
		grad_f.append( K.function( [model.input], K.gradients( model.output, [model.input] ) ) )


	#print('\tnumber of meters = '+str(len(list_meters[0])))

	# construnct the initial condition
	bias0 = np.zeros((hist_samples, 1))

	# calculate the impact on the test data
	x_0 = update_data(list_meters, t_0_test, t_f_test,  bias0, [0])

	hat_y = get_prediction(x_0, models)

	# clear the current session
	K.clear_session()
	del model, models

	return y_test, hat_y



# error metric
def MAE(y, hat_y):
	return np.mean( abs( y.reshape(-1) - hat_y.reshape(-1) ) )



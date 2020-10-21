# functions used to train the predictors 
from forecast_lib import *


# model without dropout
def createLSTM(X_data):
	# design network
	model = Sequential()

	model.add(LSTM(150, input_shape=(X_data.shape[1], X_data.shape[2]), return_sequences=True))

	# layer 2: LSTM
	model.add(LSTM(150, return_sequences=True))

	# layer 3: LSTM
	model.add(LSTM(150, return_sequences=False))

	model.add(Dense(output_dim=200, activation='relu'))

	model.add(Dense(output_dim=100, activation='relu'))

	# layer 4: dense
	model.add(Dense(1))

	model.compile(loss=loss_f, optimizer='adadelta')
	return model

# model with dropout
def createLSTM_dropout(X_data):
	# design network
	model = Sequential()

	model.add(LSTM(150, input_shape=(X_data.shape[1], X_data.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))

	# layer 2: LSTM
	model.add(LSTM(150, return_sequences=True))
	model.add(Dropout(0.2))

	# layer 3: LSTM
	model.add(LSTM(150, return_sequences=False))
	model.add(Dropout(0.2))

	model.add(Dense(output_dim=200, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(output_dim=100, activation='relu'))
	model.add(Dropout(0.2))

	# layer 4: dense
	model.add(Dense(1))

	model.compile(loss=loss_f, optimizer='adadelta')
	return model


def createLSTM_dropout_defense(X_data, rho_d):
	# design network
	model = Sequential()

	model.add(Dropout(1-rho_d, input_shape=(X_data.shape[1], X_data.shape[2]) ) )

	model.add(LSTM(150, return_sequences=True))

	# layer 2: LSTM
	model.add(LSTM(150, return_sequences=True))

	# layer 3: LSTM
	model.add(LSTM(150, return_sequences=False))

	model.add(Dense(output_dim=200, activation='relu'))

	model.add(Dense(output_dim=100, activation='relu'))

	# layer 4: dense
	model.add(Dense(1))

	model.compile(loss=loss_f, optimizer='adadelta')
	return model



def part_data(t_0, t_f, meters):
	raw_data = load_meters[t_0:t_f, meters]
	raw_temp = temp[t_0:t_f]
	total_samples = t_f-t_0-hist_samples
	features = np.zeros((total_samples, hist_samples, len(meters)+1))
	for t in range(total_samples):
		load_t = raw_data[t : t + hist_samples, :]
		temp_t = raw_temp[t : t + hist_samples]
		#pdb.set_trace()
		features[t, :, :] = np.concatenate((load_t, temp_t.reshape(-1,1)), axis=1)
	return features


def extract_data(list_meters):
	num_sets = len(list_meters)

	input_data_train = []
	input_data_test = []
	input_data_test_a = []
	for k in range(num_sets):
		meters_k = list_meters[k]
		input_data_train.append( part_data(0, train_samples, meters_k) )
		input_data_test_a.append( part_data(train_samples, train_samples+test_samples_a, meters_k) )
		input_data_test.append( part_data(train_samples+test_samples_a, T, meters_k) )

	y_train = y[hist_samples : train_samples]
	y_test_a = y[hist_samples + train_samples : train_samples + test_samples_a]
	y_test = y[hist_samples + train_samples + test_samples_a : T]

	#pdb.set_trace()
	data = {'x_train': input_data_train, 'x_test_a': input_data_test_a, 'x_test': input_data_test, 'y_train': y_train, 'y_test_a': y_test_a, 'y_test': y_test, 'meters': list_meters}

	return data





def train_and_save_forecaster(directory, data, name, rho_d=1):
	# train the first forecast
	x_train = data['x_train']
	y_train = data['y_train']

	#print(loss_f)

	n = len(x_train)
	for i in range(n): 

		# start a session
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		K.set_session(sess)

		if rho_d<1:
			dropout=True
		else:
			dropout=False

		if dropout:
			if rho_d < 0:
				model_i = createLSTM_dropout(x_train[i])
			else:
				model_i = createLSTM_dropout_defense(x_train[i], rho_d)
		else:
			model_i = createLSTM(x_train[i])

		hist_i = model_i.fit(x_train[i], y_train, epochs=75, verbose=0, validation_split=.2, shuffle=False, callbacks=[early_stopping])

		if n>1:
			model_name =  name + '_' + str(i)
		else:
			model_name =  name 
		model_i.save(directory + model_name + '.h5')

		np.save(directory + 'hist_'+model_name+'.npy', hist_i.history)

		# clear the current session
		K.clear_session()
	return



def get_partition(m, n):
	elements = random.sample( set(range( m )), m )
	size_set = int(m/n)
	list_sets = []
	for i in range(n-1):
		set_i = elements[ i*size_set : (1+i)*size_set ]
		list_sets.append( set_i )
	set_i = elements[ (n-1)*size_set : ]
	list_sets.append( set_i )
	return list_sets

def get_sets(m, n):
	partition = get_partition(m, n)
	list_meters = []
	for i in range(n):
		set_i = []
		members_i = set(range(n))-set([i])
		for j in members_i:
			set_i += partition[j]
		list_meters.append(set_i)
	return list_meters



'''
def prune_forecaster(directory, data, name):
	loaded_model = tf.keras.models.load_model(dir_rep + model_name)

	epochs = 75
	end_step = epochs

	new_pruning_params = {
	      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
		                                           final_sparsity=0.90,
		                                           begin_step=0,
		                                           end_step=end_step,
		                                           frequency=100)
	}

	new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
	new_pruned_model.summary()

	new_pruned_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])


	# Add a pruning step callback to peg the pruning step to the optimizer's
	# step. Also add a callback to add pruning summaries to tensorboard
	callbacks = [
	    sparsity.UpdatePruningStep(),
	    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
	]

	new_pruned_model.fit(x_train, y_train,
		  batch_size=batch_size,
		  callbacks=callbacks,
		  validation_split=.2)

	
	score = new_pruned_model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


	final_model = sparsity.strip_pruning(pruned_model)
	final_model.summary()

	return
'''

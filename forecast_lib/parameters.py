# parameters of the system

# data files
temp = np.load('./data/temp_norm.npy')
y = np.load('./data/total_load_norm.npy')
load_meters = np.load('./data/load_meters_norm.npy')


# system parameters
T, num_meters = load_meters.shape
num_samples = T


# training parameters
hist_samples = 24

train_samples = int(0.8 * num_samples)
test_samples_a = int(0.1 * num_samples)

use_mse = False

if use_mse:
	loss_f = 'mean_squared_error'
	loss = 'mse'
else:
	loss_f = 'mean_absolute_error'
	loss = 'mae'

early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=False, min_delta=0.001)



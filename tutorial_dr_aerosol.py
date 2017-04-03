#### Tutorial Code for performing Phase Neural Network or Fourier Neural Network for distributional regression ####
import numpy as np
import aux_fct
import phase_fourier_dr_nn
from math import sqrt
# State your path for the Aerosol Dataset MISR1
path = '/homes/hlaw/Files/Fourier-Phase-Neural-Network'
# Load Dataset into features and labels
misr_data_x, misr_data_y = aux_fct.load_data(path)
variance = np.var(np.concatenate(misr_data_x), axis = 0) # For calculating signal to noise ratio

# State Your parameters for the neural network
learning_rate = 0.3
reg_1 = 10.0  # L2 Regularisation for frequencies layer 
reg_2 = 0.01 # L2 Regularisation for output layer
n_freq = 60 # Number of frequencies to use
batch_size = 20
no_epochs = 100
noise_ratio_t = 0.0 # Signal to Noise Ratio added to test set only (varying noises for varying bags)
version = 'Phase' # To perform fourier neural network, use 'Fourier'
n_cpu = 1 # Number of CPUs available

# Construct Train and Test Set
x_train = misr_data_x[:640,:,:]
y_train = misr_data_y[:640]
x_test = misr_data_x[640:,:,:]
y_test = misr_data_y[640:]

# Compute the median heuristic for the RBF kernel bandwidth as initialisation.
width_x = aux_fct.get_sigma_median_heuristic(np.concatenate(x_train))
init_sd = 1.0/width_x

# Add noise to test set
x_test_n = np.zeros((160, 100, 16))
latent_t = noise_ratio_t * variance * np.random.uniform(low = 0.0, high = 1.0, size = (160, 1, 16))
for i in range(160):
	for k in range(16):
		x_test_n[i,:,k] = x_test[i,:,k] + sqrt(latent_t[i,:,k]) * np.random.normal(size = (100))

# Run network with the setup as below
accuracy = phase_fourier_dr_nn.phase_fourier_nn(x_train, y_train, x_test_n, y_test, n_freq, learning_rate, reg_1, reg_2, batch_size, no_epochs, version, init_sd, n_cpu)
baseline = np.sqrt(np.mean(np.square(y_test - np.mean(y_test)))) # Using average as baseline
print( 'Baseline Mean accuracy: ',str(baseline) )
print( 'Noise Level added to test set:', noise_ratio_t)
print( version, 'Network RMSE accuracy: ', str(sqrt(accuracy)) )

# Fourier has learnt to be somewhat invariant to noise on original dataset, while Phase is directly invariant to noise, hence is slightly more robust.


#### Tutorial Code for performing Phase Neural Network or Fourier Neural Network for distributional regression ####
import numpy as np
import aux_fct
import phase_fourier_dr_nn
from math import sqrt
# State your path for the Aerosol Dataset MISR1
path = '/homes/hlaw/Files/phase_discrepancies/MIR_datasets'
# Load Dataset into features and labels
misr_data_x, misr_data_y = aux_fct.load_data(path)

# State Your parameters for the neural network
learning_rate = 0.3
reg_1 = 10.0  # L2 Regularisation for frequencies layer 
reg_2 = 0.001 # L2 Regularisation for output layer
n_freq = 60 # Number of frequencies to use
batch_size = 10
no_epochs = 120
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

accuracy = phase_fourier_dr_nn.phase_fourier_nn(x_train, y_train, x_test, y_test, n_freq, learning_rate, reg_1, reg_2, batch_size, no_epochs, version, init_sd, n_cpu)

print( 'RMSE accuracy: ' + str(sqrt(accuracy)) )

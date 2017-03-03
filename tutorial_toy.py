import numpy as np
import aux_fct
import phase_fourier_nn
from math import sqrt

num_bags = 1000
dim = 5
df0=4
df1=8
df=np.random.choice([df0,df1],num_bags)
labels_a=np.array(1*(df==df1))
num1=np.count_nonzero(labels_a)
X=np.zeros((num_bags,dim))
X[df==df1,:] = np.random.chisquare(df=df1,size=(num1,dim))
X[df==df0,:] = np.random.chisquare(df=df0,size=(num_bags-num1,dim))

labels = np.zeros((num_bags, 2))
labels[:,0] = np.array(1*(df==df0))
labels[:,1] = np.array(1*(df==df1))

learning_rate = 0.3
reg = 0.1
n_freq = 60 # Number of frequencies to use
batch_size = 10
no_epochs = 40
n_cpu = 1 # Number of CPUs available

# Construct Train and Test Set
x_train = X[:750]
y_train = labels[:750]
x_test = X[750:]
y_test = labels[750:]

width_x = aux_fct.get_sigma_median_heuristic(np.concatenate(x_train))
init_sd = 1.0/width_x

accuracy = fourier_nn.fourier_nn(x_train, y_train, x_test, y_test, n_freq, learning_rate, reg, batch_size, no_epochs, init_sd, n_cpu)



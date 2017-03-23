### Phase Fourier Network for distrbutional regression ###
import numpy as np
import tensorflow as tf
import kernel_layer
import math

def phase_fourier_nn(x_train, y_train, x_test, y_test, n_hidden, lr, reg_1, reg_2, batch_size, no_epochs, version, init_sd, n_cpu):
    decay = lr/no_epochs # Using a particular learning rate decay schedule
    shape = x_train.shape 
    assert len(shape) == 3
    num_bags = shape[0]
    bag_size = shape[1]
    dim = shape[2]
    n_vars = y_train.shape[1]
    epsilon = 1e-3 # For batch normalisation
    # Setup input variables
    x = tf.placeholder("float", [None, bag_size, dim])
    y = tf.placeholder("float", [None, n_vars])
    lr_var = tf.placeholder(tf.float32, shape=[])
    mean_batch = tf.placeholder("float",[n_hidden * 2])
    var_batch = tf.placeholder("float",[n_hidden * 2])
    training = tf.placeholder(tf.bool, shape=[]) # True if Training, False in testing, for batch normalisation
    # Store layer weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([dim, n_hidden], stddev = init_sd )),
        'out': tf.Variable(tf.random_normal([n_hidden * 2, n_vars] ))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_vars]))
    }
    # Set up network
    if version == 'Fourier':
        output, mean_b, var_b = kernel_layer.kernel_nn(x, weights, n_hidden, epsilon, mean_batch, var_batch, bias = biases, train = training, rescale = True, dr = True, norm = False)
    elif version == 'Phase':
        output, mean_b, var_b = kernel_layer.kernel_nn(x, weights, n_hidden, epsilon, mean_batch, var_batch, bias = biases, train = training, rescale = True, dr = True, norm = True)
    else:
        raise Exception('Error in version name, Either Fourier or Phase available')
    # Define loss function and optimizer
    y_size = tf.cast( tf.shape(y)[0],'float32')
    cost = tf.div( 2 * tf.nn.l2_loss(output - y), y_size) + reg_1 * tf.nn.l2_loss(weights['h1']) + reg_2 * tf.nn.l2_loss(weights['out'])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_var).minimize(cost) # ADAM optimizer with particular learning rate
    #optimizer = tf.train.GradientDescentOptimizer(lr_var).minimize(cost)
    # Initializing the variables
    init = tf.global_variables_initializer()
    mean_avg = np.zeros(n_hidden * 2)
    var_avg = np.zeros(n_hidden * 2)
    # Parallisation
    config = tf.ConfigProto(intra_op_parallelism_threads=n_cpu, inter_op_parallelism_threads=n_cpu)
    display_step = 10
    with tf.Session(config = config) as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(no_epochs):
            lr = lr * 1/(1 + decay * epoch)
            avg_cost = 0.0
            total_batch = int(float(num_bags)/batch_size)
            # Loop over all batches
            permu_list = np.random.choice(num_bags, num_bags)
            for i in range(total_batch):
                permu = permu_list[batch_size*i : batch_size * (i +1)]
                batch_x = x_train[permu, :]
                batch_y = y_train[permu,:]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, m_b,v_b = sess.run([optimizer, cost, mean_b, var_b], feed_dict={x: batch_x,
                                                                                      y: batch_y,
                                                                                      lr_var: lr,
                                                                                      training: True,
                                                                                      mean_batch: mean_avg,  
                                                                                      var_batch: var_avg })
                avg_cost += c/total_batch
                # Store mean and variance of batches for testing
                if i == 0:
                    mean_b_avg = m_b/total_batch
                    var_b_avg = v_b/total_batch
                else:
                    mean_b_avg = mean_b_avg + m_b/total_batch
                    var_b_avg = var_b_avg + v_b/total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch +1), "cost=", \
                "{:.9}".format(avg_cost))
        test_accuracy = tf.reduce_mean(tf.square(output - y))
        result = test_accuracy.eval(feed_dict={x: x_test, y: y_test, training: False, mean_batch: mean_b_avg, var_batch: var_b_avg})
    return result

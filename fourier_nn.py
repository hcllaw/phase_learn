### Fourier Neural Network for Classification ###
import numpy as np
import tensorflow as tf
import kernel_layer
import math

def fourier_nn(x_train, y_train, x_test, y_test, n_hidden, lr, reg, batch_size, no_epochs, init_sd, n_cpu):
    decay = lr/no_epochs # Using a particular learning rate decay schedule
    shape = x_train.shape 
    assert len(shape) == 2
    num_bags = shape[0]
    dim = shape[1]
    n_vars = y_train.shape[1]
    epsilon = 1e-3 # For batch normalisation
    # Setup input variables
    x = tf.placeholder("float", [None, dim])
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
    output, mean_b, var_b = kernel_layer.kernel_nn(x, weights, n_hidden, epsilon, mean_batch, var_batch, bias = biases, train = training, rescale = True, dr = False, norm = False)
    # Define loss function and optimizer
    y_size = tf.cast( tf.shape(y)[0],'float32')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)) + reg * tf.nn.l2_loss(weights['h1']) + reg * tf.nn.l2_loss(weights['out'])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_var).minimize(cost) # ADAM optimizer with particular learning rate
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
            total_batch = int(num_bags/batch_size)
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
        correct_prediction_t = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        test_accuracy = tf.reduce_mean(tf.cast(correct_prediction_t, "float"))
        result = test_accuracy.eval(feed_dict={x: x_test, y: y_test, training: False, mean_batch: mean_b_avg, var_batch: var_b_avg})
    return result

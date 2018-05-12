import tensorflow as tf
import tensorflow.contrib.layers as tflayers

# Initializers
sigma = 1.0
weight_initializer = tflayers.variance_scaling_initializer(mode="FAN_AVG", uniform=True, factor=sigma)
bias_initializer = tf.zeros_initializer()


def mlp_input_layer(n_inputs):
    return tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])


def hidden_layer(inputs, n_input, n_neurons):
    # Variables for hidden weights and biases
    weight_hidden = tf.Variable(weight_initializer([n_input, n_neurons]))
    bias_hidden = tf.Variable(bias_initializer([n_neurons]))

    # Hidden layer
    return tf.nn.relu(tf.add(tf.matmul(inputs, weight_hidden), bias_hidden))


def mlp_output_layer(inputs, n_inputs, n_output):
    weight_out = tf.Variable(weight_initializer([n_inputs, n_output]))
    bias_out = tf.Variable(bias_initializer([n_output]))
    return tf.transpose(tf.add(tf.matmul(inputs, weight_out), bias_out))


def mlp_cost_function(actual, expected):
    return tf.reduce_mean(tf.squared_difference(actual, expected))


def cnn_input1d_layer(n_history, n_features):
    return tf.placeholder(tf.float32, [None, n_history, n_features])


def cnn_conv1d_layer(inputs, output_dim, conv_w=9, conv_s=2, padding="SAME", name="conv1d", stddev=0.02):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [conv_w, inputs.get_shape().as_list()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        c = tf.nn.conv1d(inputs, w, conv_s, padding=padding)
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        return c + b


def cnn_lrelu_layer(inputs):
    lrelu_factor = 0.2
    with tf.name_scope("lrelu"):
        x = tf.identity(inputs)
        return (0.5 * (1 + lrelu_factor)) * x + (0.5 * (1 - lrelu_factor)) * tf.abs(x)


def cnn_relu_layer(inputs):
    return tf.nn.relu(inputs)


def cnn_pool2d_layer(inputs, size, strides):
    return tf.layers.max_pooling2d(inputs=inputs, pool_size=size, strides=strides)


def cnn_input2d_layer(batch_size, n_data, n_history, n_features):
    return tf.reshape(tf.cast(batch_size, tf.float32), [-1, n_data, n_history, n_features])


def cnn_conv2d_layer(inputs, n_filter, kernel):
    return tf.layers.conv2d(
            inputs=inputs,
            filters=n_filter,
            kernel_size=kernel,
            padding="same",
            activation=tf.nn.relu)


def cnn_batchnorm1d_layer(inputs):
    with tf.variable_scope("batchnorm"):
        inputs = tf.identity(inputs)
        channels = inputs.get_shape()[-1]
        offset = tf.get_variable("offset", [channels],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(inputs, axes=[0, 1], keep_dims=False)

        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(inputs, mean, variance, offset, scale,
                                               variance_epsilon=variance_epsilon)
        return normalized


def cnn_dropout_layer(inputs, dropout):
    return tf.nn.dropout(inputs, keep_prob=1 - dropout)


def training_component(inputs):
    return tf.train.AdamOptimizer().minimize(inputs)

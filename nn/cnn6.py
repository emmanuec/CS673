import functools as ft
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from nn.model import Model
import nn.nnutils.layer_builder as build
import util.data_extract as d_extract


def accuracy(actual, expected):
    return (100.0 * np.sum(np.argmax(actual, 1) == np.argmax(expected, 1))
            / actual.shape[0])


class CNN6(Model):

    def __init__(self, file_name):
        self.file_name = file_name
        # Model architecture parameters - Part 1
        self.n_history = 15  # in days
        n_features = 17  # 1 for each stock for current data
        dropout = 0.5
        filter_num = 8
        conv_layer_specs = [
            (filter_num * 2, dropout),
            (filter_num * 4, dropout),
            (filter_num * 8, dropout),
            (filter_num * 8, dropout),
            (filter_num * 8, dropout)
        ]

        # Placeholder
        self.input_values = build.cnn_input1d_layer(self.n_history, n_features)
        self.expected_values = tf.placeholder(dtype=tf.float32, shape=[None])

        # **Convolutional Layers**

        conv_layers = []
        conv_layer_1 = build.cnn_conv1d_layer(self.input_values, filter_num)
        conv_layers_output = build.cnn_relu_layer(conv_layer_1)
        conv_layers.append(conv_layers_output)

        for _, (out_channels, dropout) in enumerate(conv_layer_specs):
            with tf.variable_scope("conv_%d" % (len(conv_layers) + 1)):
                rectified_layer = build.cnn_lrelu_layer(conv_layers[-1])

                # Convolutional Layer
                # [batch, in_width, in_channels] => [batch, in_width/2, out_channels]
                conv_layer = build.cnn_conv1d_layer(rectified_layer, out_channels)

                conv_layers.append(conv_layer)

        # Adjust Convolutional Layer output for Fully Connected Layer
        n_conv_layers_output_adjust = ft.reduce(lambda x, y: x*y, conv_layers[-1].get_shape().as_list()[1:])
        conv_layers_output_adjust = tf.reshape(conv_layers[-1], [-1, n_conv_layers_output_adjust])

        # Model architecture parameters - Part 2
        n_neurons_1 = 1024
        n_neurons_2 = 512
        n_neurons_3 = 256
        n_neurons_4 = 128
        n_output = 1

        # **Fully Connected Layer**
        fcc1_hidden_layer_1 = build.hidden_layer(conv_layers_output_adjust, n_conv_layers_output_adjust, n_neurons_1)
        fcc1_hidden_layer_2 = build.hidden_layer(fcc1_hidden_layer_1, n_neurons_1, n_neurons_2)
        fcc1_hidden_layer_3 = build.hidden_layer(fcc1_hidden_layer_2, n_neurons_2, n_neurons_3)
        fcc1_hidden_layer_4 = build.hidden_layer(fcc1_hidden_layer_3, n_neurons_3, n_neurons_4)


        fcc_dropout_layer = build.cnn_dropout_layer(fcc1_hidden_layer_4, dropout)

        # Hidden layers
        hidden_layer_1 = build.hidden_layer(fcc_dropout_layer, n_neurons_4, n_neurons_1)
        hidden_layer_2 = build.hidden_layer(hidden_layer_1, n_neurons_1, n_neurons_2)
        hidden_layer_3 = build.hidden_layer(hidden_layer_2, n_neurons_2, n_neurons_3)
        hidden_layer_4 = build.hidden_layer(hidden_layer_3, n_neurons_3, n_neurons_4)
        # Output layer
        self.output_layer = build.mlp_output_layer(hidden_layer_4, n_neurons_4, n_output)

        # Cost function
        self.mse = build.regression_cost_function(self.output_layer, self.expected_values)

        # Optimizer
        self.opt = build.adam_opt_training_component(self.mse, 0.0001)

    def train(self):
        # Set up data
        data_set = d_extract.get_time_data(self.n_history, self.file_name)
        input_train = data_set.input_train
        output_train = data_set.output_train
        input_test = data_set.input_test
        output_test = data_set.output_test

        # Make Session
        net = tf.Session()

        # Run initializer
        net.run(tf.global_variables_initializer())

        # Setup interactive plot
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(output_test)
        line2, = ax1.plot(output_test * 0.5)
        ax1.set_title("Prediction vs Actual")
        ax1.set_ylabel("Closing Price")
        ax1.set_xlabel("Date")
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.show()

        # Number of epochs and batch size
        epochs = 500
        batch_size = 50

        best_mse = 1
        best_epoch = 1
        best_batch = 1

        for e in range(epochs):

            # Minibatch training
            for i in range(0, len(output_train) // batch_size):
                start = i * batch_size
                batch_x = input_train[start:start + batch_size]
                batch_y = output_train[start:start + batch_size]
                # Run optimizer with batch
                net.run(self.opt, feed_dict={self.input_values: batch_x, self.expected_values: batch_y})

                # Show progress
                if np.mod(i, 5) == 0:
                    # Prediction
                    pred = net.run(self.output_layer, feed_dict={self.input_values: input_test})
                    curr_mse = net.run(self.mse,
                                       feed_dict={self.input_values: input_test, self.expected_values: output_test})
                    if curr_mse < best_mse:
                        best_mse = curr_mse
                        best_epoch = e
                        best_batch = i
                    line2.set_ydata(pred)
                    plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                    file_name = "D:\\Files\\Box Sync\\classes\\Spring2018\\CS673\\final\\data\\img/CNN7_" + str(self.file_name) + "_epoch_" + str(e) + '_batch_' + str(i) + '.pdf'
                    plt.savefig(file_name)
                    plt.pause(0.01)
        # Print final MSE after Training
        mse_final = net.run(self.mse, feed_dict={self.input_values: input_test, self.expected_values: output_test})
        print(mse_final)
        print("File Name: " + str(self.file_name))
        print("Best MSE: " + str(best_mse))
        print("Best EPOCH: " + str(best_epoch))
        print("Best BATCH: " + str(best_batch))
        tf.reset_default_graph()

    def predict(self):
        pass




import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from nn.model import Model
import nn.nnutils.layer_builder as build
import util.data_extract as d_extract


class LSTM(Model):

    def __init__(self, file_name):
        self.file_name = file_name
        self.n_history = 20
        n_features = 17
        n_lstm_layers = 4
        lstm_size = 128
        dropout = 0.75
        n_output = 1

        self.input_values = build.lstm_input_layer(self.n_history, n_features)
        self.expected_values = tf.placeholder(dtype=tf.float32, shape=[None])

        lstm_hidden_layers = build.lstm_hidden_layers(self.input_values, n_lstm_layers, lstm_size, dropout)

        transpose_layer = build.lstm_transpose(lstm_hidden_layers)

        # Output layer
        self.output_layer = build.lstm_output_layer(transpose_layer, lstm_size, n_output)

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

            # Shuffle training util
           # shuffle_indices = np.random.permutation(np.arange(len(output_train)))
           # input_train = input_train[shuffle_indices]
           # output_train = output_train[shuffle_indices]

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
                    file_name = "D:\\Files\\Box Sync\\classes\\Spring2018\\CS673\\final\\data\\img/LSTM_" + str(self.file_name) + "_epoch_" + str(e) + '_batch_' + str(i) + '.pdf'
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

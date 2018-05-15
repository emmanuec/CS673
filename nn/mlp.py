import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from nn.model import Model
import nn.nnutils.layer_builder as build
import util.data_extract as d_extract


class MLP(Model):

    def __init__(self, file_name):
        self.file_name = file_name
        # Model architecture parameters
        n_features = 17
        n_neurons_1 = 1024
        n_neurons_2 = 512
        n_neurons_3 = 256
        n_neurons_4 = 128
        #n_neurons_3 = 50
        #n_neurons_4 = 25
        n_output = 1

        # Placeholder
        self.input_values = build.mlp_input_layer(n_features)
        self.expected_values = tf.placeholder(dtype=tf.float32, shape=[None])

        # Hidden layers
        hidden_layer_1 = build.hidden_layer(self.input_values, n_features, n_neurons_1)
        hidden_layer_2 = build.hidden_layer(hidden_layer_1, n_neurons_1, n_neurons_2)
        hidden_layer_3 = build.hidden_layer(hidden_layer_2, n_neurons_2, n_neurons_3)
        #hidden_layer_3 = build.hidden_layer(self.input_values, n_stocks, n_neurons_3)
        hidden_layer_4 = build.hidden_layer(hidden_layer_3, n_neurons_3, n_neurons_4)

        # Output layer
        self.output_layer = build.mlp_output_layer(hidden_layer_4, n_neurons_4, n_output)

        # Cost function
        self.mse = build.regression_cost_function(self.output_layer, self.expected_values)

        # Optimizer
        self.opt = build.adam_opt_training_component(self.mse)

    def train(self):
        # Set up data
        data_set = d_extract.get_data(self.file_name)
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
            shuffle_indices = np.random.permutation(np.arange(len(output_train)))
            input_train = input_train[shuffle_indices]
            output_train = output_train[shuffle_indices]

            # Minibatch training
            for i in range(0, len(output_train) // batch_size):
                start = i * batch_size
                batch_x = input_train[start:start + batch_size]
                batch_y = output_train[start:start + batch_size]
                # Run optimizer with batch
                net.run(self.opt, feed_dict={self.input_values: batch_x, self.expected_values: batch_y})

                # Show progress
                if np.mod(i, 10) == 0:
                    # Prediction
                    pred = net.run(self.output_layer, feed_dict={self.input_values: input_test})
                    curr_mse = net.run(self.mse, feed_dict={self.input_values: input_test, self.expected_values: output_test})
                    if curr_mse < best_mse:
                        best_mse = curr_mse
                        best_epoch = e
                        best_batch = i
                    line2.set_ydata(pred)
                    plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                    file_name = "D:\\Files\\Box Sync\\classes\\Spring2018\\CS673\\final\\data\\img/mlp_" + str(self.file_name) + "_epoch_" + str(e) + '_batch_' + str(i) + '.pdf'
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

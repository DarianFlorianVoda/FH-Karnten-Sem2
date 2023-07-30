import random

import numpy as np

from FeedForwardNetwork.Layer import Layer


class MultiLayerPerceptron():
    """
        A Multi-Layer Perceptron (MLP) class for implementing a neural network with one hidden layer
        and an output layer that supports both binary and multiclass classification.

        Attributes:
            learning_rate (float): The learning rate for the training process. Default is 0.01.
            num_iteration (int): The number of training iterations. Default is 100.
            weight_decay (float): The weight decay coefficient for regularization. Default is 0.0.
            layers (list): A list to store the layers of the neural network.

        Methods:
            add_output_layer(num_neurons, activation='sigmoid'): Add the output layer to the architecture.
            add_hidden_layer(num_neuron, activation='sigmoid'): Add a hidden layer to the architecture.
            update_layers(target): Update all the layers by calculating the updated weights and then
                                   updating the weights all at once when the new weights are found.
            fit(X, y): Main training function of the neural network algorithm using stochastic gradient descent.
            predict(row): Prediction function that will take a row of input and give back the output of
                          the whole neural network. Supports both binary and multiclass classification.
            accuracy_multiclass(X_test, y_test): Calculate the accuracy of the trained neural network on
                                                multiclass classification tasks using test data.
        """
    def __init__(self, learning_rate=0.01, num_iteration=100, weight_decay=0.0):
        """
                Constructor for MultiLayerPerceptron class.

                Parameters:
                    learning_rate (float, optional): The learning rate for the training process. Default is 0.01.
                    num_iteration (int, optional): The number of training iterations. Default is 100.
                    weight_decay (float, optional): The weight decay coefficient for regularization. Default is 0.0.
        """
        # Layers
        self.layers = []

        # Training parameters
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.weight_decay = weight_decay

    def add_output_layer(self, num_neurons, activation='sigmoid'):
        """
                Add the output layer to the architecture.

                Parameters:
                    num_neurons (int): The number of neurons in the output layer.
                    activation (str, optional): The activation function to use for the output layer.
                                               Default is 'sigmoid'.
        """
        self.layers.insert(0, Layer(num_neuron=num_neurons, activation=activation, is_output_layer=True))

    def add_hidden_layer(self, num_neuron, activation='sigmoid'):
        """
                Add a hidden layer to the architecture.

                Parameters:
                    num_neuron (int): The number of neurons in the hidden layer.
                    activation (str, optional): The activation function to use for the hidden layer.
                                               Default is 'sigmoid'.
        """
        # Create a hidden layer
        hidden_layer = Layer(num_neuron, activation=activation, position_in_layer=len(self.layers))
        # Attach the last added layer to this new layer
        hidden_layer.attach(self.layers[0])
        # Add this layer to the architecture
        self.activation = activation
        self.layers.insert(0, hidden_layer)

    def update_layers(self, target):
        """
                Update all the layers by calculating the updated weights and then updating the weights
                all at once when the new weights are found.

                Parameters:
                    target (float): The target output value for the neural network during training.
        """
        # Iterate over each of the layer in reverse order
        # to calculate the updated weights
        for layer in reversed(self.layers):
            # Calculate update the hidden layer
            for neuron in layer.neurons:
                neuron.calculate_update(self.learning_rate, target, self.weight_decay)

        # Iterate over each of the layer in normal order
        # to update the weights
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_neuron()

    def fit(self, X, y):
        """
               Main training function of the neural network algorithm using stochastic gradient descent.

               Parameters:
                   X (numpy.ndarray): The input training data.
                   y (numpy.ndarray): The target labels for the training data.
        """
        num_row = len(X)
        num_feature = len(X[0])  # Here we assume that we have a rectangular matrix

        # Init the weights throughout each of the layer
        self.layers[0].init_layer(num_feature)

        for i in range(1, len(self.layers)):
            num_input = len(self.layers[i - 1].neurons)
            self.layers[i].init_layer(num_input)

        # Launch the training algorithm
        for i in range(self.num_iteration):

            # Stochastic Gradient Descent
            r_i = random.randint(0, num_row - 1)
            row = X[r_i]  # take the random sample from the dataset
            yhat = self.predict(row)
            target = y[r_i]

            # Update the layers using backpropagation
            self.update_layers(target)

            # At every 100 iteration we calculate the error
            # on the whole training set
            if i % 1000 == 0:
                total_error = 0
                for r_i in range(num_row):
                    row = X[r_i]
                    yhat = self.predict(row)
                    error = (y[r_i] - yhat)
                    total_error = total_error + error ** 2
                mean_error = total_error / num_row
                print(f"Iteration {i} with error = {mean_error}")

    def predict(self, row):
        """
                Prediction function that will take a row of input and give back the output of the whole neural network.

                Parameters:
                    row (numpy.ndarray): A row of input data for prediction.

                Returns:
                    float or numpy.ndarray: The output of the neural network after applying the activation function.
                                            For binary classification, the output is a float (0 or 1).
                                            For multiclass classification, the output is a numpy array representing
                                            the probabilities for each class label.
        """
        # Gather all the activation in the hidden layer
        activations = self.layers[0].predict(row)
        for i in range(1, len(self.layers)):
            activations = self.layers[i].predict(activations)

        # Output layer's activation function
        if len(self.layers[0].neurons) == 1:
            # For binary classification, use sigmoid activation for the output layer
            exp_activations = np.exp(activations)
            probabilities = exp_activations / (1 + exp_activations)
            return probabilities[0]
        else:
            # For multiclass classification, use softmax activation for the output layer
            exp_activations = np.exp(activations - np.max(activations))  # To prevent numerical instability
            probabilities = exp_activations / np.sum(exp_activations)
            return probabilities

    def accuracy_multiclass(self, X_test, y_test):
        """
                Calculate the accuracy of the trained neural network on multiclass classification tasks using test data.

                Parameters:
                    X_test (numpy.ndarray): The input test data.
                    y_test (numpy.ndarray): The true target labels for the test data.

                Returns:
                    float: The accuracy of the neural network on the multiclass classification task.
        """
        correct = 0
        total = len(y_test)

        for i, row in enumerate(X_test):
            y_pred = np.argmax(self.predict(row))
            if y_pred == y_test[i]:
                correct += 1

        return correct / total
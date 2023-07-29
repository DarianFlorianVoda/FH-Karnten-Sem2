import random

from FeedForwardNetwork.Layer import Layer


class MultiLayerPerceptron():
    '''
        We will be creating the multi-layer perceptron with only two layer:
        an input layer, a perceptrons layer and a one neuron output layer which does binary classification
    '''

    def __init__(self, learning_rate=0.01, num_iteration=100, weight_decay=0.0):

        # Layers
        self.layers = []

        # Training parameters
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.weight_decay = weight_decay

    def add_output_layer(self, num_neuron):
        '''
            This helper function will create a new output layer and add it to the architecture
        '''
        self.layers.insert(0, Layer(num_neuron, is_output_layer=True))

    def add_hidden_layer(self, num_neuron):
        '''
            This helper function will create a new hidden layer, add it to the architecture
            and finally attach it to the front of the architecture
        '''
        # Create an hidden layer
        hidden_layer = Layer(num_neuron)
        # Attach the last added layer to this new layer
        hidden_layer.attach(self.layers[0])
        # Add this layers to the architecture
        self.layers.insert(0, hidden_layer)

    def update_layers(self, target):
        '''
            Will update all the layers by calculating the updated weights and then updating
            the weights all at once when the new weights are found.
        '''
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
        '''
            Main training function of the neural network algorithm. This will make use of backpropagation.
            It will use stochastic gradient descent by selecting one row at random from the dataset and
            use predict to calculate the error. The error will then be backpropagated and new weights calculated.
            Once all the new weights are calculated, the whole network weights will be updated
        '''
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
        '''
            Prediction function that will take a row of input and give back the output
            of the whole neural network.
        '''

        # Gather all the activation in the hidden layer

        activations = self.layers[0].predict(row)
        for i in range(1, len(self.layers)):
            activations = self.layers[i].predict(activations)

        outputs = []
        for activation in activations:
            # Decide if we output a 1 or 0
            if activation >= 0.5:
                outputs.append(1.0)
            else:
                outputs.append(0.0)

        # We currently have only One output allowed
        return outputs[0]
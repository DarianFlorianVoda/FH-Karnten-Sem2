import random
import math


class Neuron():
    '''
        A conceptual Neuron hat can be trained using a
        fit and predict methodology, without any library
    '''

    def __init__(self, position_in_layer, is_output_neuron=False):
        self.weights = []
        self.inputs = []
        self.output = None

        # This is used for the backpropagation update
        self.updated_weights = []
        # This is used to know how to update the weights
        self.is_output_neuron = is_output_neuron
        # This delta is used for the update at the backpropagation
        self.delta = None
        # This is used for the backpropagation update
        self.position_in_layer = position_in_layer

    def attach_to_output(self, neurons):
        '''
            Helper function to store the reference of the other neurons
            To this particular neuron (used for backpropagation)
        '''

        self.output_neurons = neurons

    def sigmoid(self, x):
        '''
            simple sigmoid function (logistic) used for the activation
        '''
        return 1 / (1 + math.exp(-x))

    def init_weights(self, num_input):
        '''
            This is used to setup the weights when we know how many inputs there is for
            a given neuron
        '''

        # Randomly initalize the weights
        for i in range(num_input + 1):
            self.weights.append(random.uniform(0, 1))

    def predict(self, row):
        '''
            Given a row of data it will predict what the output should be for
            this given neuron. We can have many input, but only one output for a neuron
        '''

        # Reset the inputs
        self.inputs = []

        # We iterate over the weights and the features in the given row
        activation = 0
        for weight, feature in zip(self.weights, row):
            self.inputs.append(feature)
            activation = activation + weight * feature

        self.output = self.sigmoid(activation)
        return self.output

    def update_neuron(self):
        '''
            Will update a given neuron weights by replacing the current weights
            with those used during the backpropagation. This need to be done at the end of the
            backpropagation
        '''

        self.weights = []
        for new_weight in self.updated_weights:
            self.weights.append(new_weight)

    def calculate_update(self, learning_rate, target):
        '''
            This function will calculate the updated weights for this neuron. It will first calculate
            the right delta (depending if this neuron is a ouput or a hidden neuron), then it will
            calculate the right updated_weights. It will not overwrite the weights yet as they are needed
            for other update in the backpropagation algorithm.
        '''

        if self.is_output_neuron:
            # Calculate the delta for the output
            self.delta = (self.output - target) * self.output * (1 - self.output)
        else:
            # Calculate the delta
            delta_sum = 0
            # this is to know which weights this neuron is contributing in the output layer
            cur_weight_index = self.position_in_layer
            for output_neuron in self.output_neurons:
                delta_sum = delta_sum + (output_neuron.delta * output_neuron.weights[cur_weight_index])

            # Update this neuron delta
            self.delta = delta_sum * self.output * (1 - self.output)

        # Reset the update weights
        self.updated_weights = []

        # Iterate over each weight and update them
        for cur_weight, cur_input in zip(self.weights, self.inputs):
            gradient = self.delta * cur_input
            new_weight = cur_weight - learning_rate * gradient
            self.updated_weights.append(new_weight)
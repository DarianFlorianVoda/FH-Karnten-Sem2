import math
import random

from FeedForwardNetwork.utils import sigmoid, relu, tanh, softmax


class Neuron():
    """
        A Neuron class that represents a conceptual neuron, which can be trained and used for prediction
        using a fit and predict methodology without any library.

        Attributes:
            position_in_layer (int): The position index of the neuron in its layer.
            is_output_neuron (bool): Indicates if the neuron is an output neuron. Default is False.

        Methods:
            attach_to_output(neurons): Helper function to store the reference of the other neurons
                                       to this particular neuron (used for backpropagation).
            init_weights(num_input): Initialize the weights of the neuron given the number of inputs.
            predict(row, activation_fun='sigmoid'): Given a row of data, predict the output of the neuron.
            update_neuron(): Update the neuron's weights with the weights calculated during backpropagation.
            calculate_update(learning_rate, target, weight_decay): Calculate the updated weights for the neuron
                                                                  using the backpropagation algorithm.

        Note:
            The activation function for the neuron can be specified when calling the predict() method.
            The options for activation_fun are: 'sigmoid', 'relu', 'tanh', or 'softmax'.
        """
    def __init__(self, position_in_layer, is_output_neuron=False):
        """
                Constructor for Neuron class.

                Parameters:
                    position_in_layer (int): The position index of the neuron in its layer.
                    is_output_neuron (bool, optional): Indicates if the neuron is an output neuron. Default is False.
        """
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
        """
                Helper function to store the reference of the other neurons to this particular neuron.
                This is used for backpropagation.

                Parameters:
                    neurons (list): The list of neurons to which this neuron will be attached.
        """
        self.output_neurons = neurons

    def init_weights(self, num_input):
        """
                Initialize the weights of the neuron given the number of inputs.

                Parameters:
                    num_input (int): The number of input features for the neuron.
        """
        # Randomly initalize the weights
        for i in range(num_input + 1):
            self.weights.append(random.uniform(0, 1))

    def predict(self, row, activation_fun='sigmoid'):
        """
                Given a row of data, predict the output of the neuron.

                Parameters:
                    row (numpy.ndarray): A row of data as input to the neuron.
                    activation_fun (str, optional): The activation function to use for the neuron's output.
                                                   Default is 'sigmoid'.

                Returns:
                    float: The output of the neuron after applying the specified activation function.
        """
        # Reset the inputs
        self.inputs = []

        # We iterate over the weights and the features in the given row
        activation = 0
        for weight, feature in zip(self.weights, row):
            self.inputs.append(feature)
            activation = activation + weight * feature

        if activation_fun == 'sigmoid':
            self.output = sigmoid(activation)
        elif activation_fun == 'relu':
            self.output = relu(activation)
        elif activation_fun == 'tanh':
            self.output = tanh(activation)
        elif activation_fun == 'softmax':
            self.output = softmax(activation)

        return self.output

    def update_neuron(self):
        """
                Update the neuron's weights with the weights calculated during backpropagation.
                This function is called at the end of the backpropagation process to apply the updates.
        """
        self.weights = []
        for new_weight in self.updated_weights:
            self.weights.append(new_weight)

    def calculate_update(self, learning_rate, target, weight_decay):
        """
                Calculate the updated weights for the neuron using the backpropagation algorithm.

                Parameters:
                    learning_rate (float): The learning rate used for weight updates during backpropagation.
                    target (float): The target output value for the neuron.
                    weight_decay (float): The weight decay parameter applied to the neuron's weights.

                Note:
                    This function should be called after the predict() method to calculate the neuron's output.

        """
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

        # Iterate over each weight and update them with weight decay
        for cur_weight, cur_input in zip(self.weights, self.inputs):
            gradient = self.delta * cur_input
            weight_decay_term = weight_decay * cur_weight
            new_weight = cur_weight - learning_rate * (gradient + weight_decay_term)
            self.updated_weights.append(new_weight)

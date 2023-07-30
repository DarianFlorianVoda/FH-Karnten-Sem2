import numpy as np

from FeedForwardNetwork.Neuron import Neuron
from FeedForwardNetwork.utils import sigmoid, relu, tanh, softmax


class Layer():
    """
        A Layer class representing a layer in a fully-connected feedforward neural network architecture.

        Attributes:
            num_neuron (int): The number of neurons in the layer.
            activation (str): The activation function to be used for the neurons in the layer. Options are:
                              'sigmoid', 'relu', 'tanh', 'softmax'.
            is_output_layer (bool): Indicates if the layer is the output layer. Default is False.
            weight_decay (float): The weight decay parameter to apply to the neurons in the layer. Default is 0.01.
            position_in_layer (int or None): The position index of the layer in the network architecture.
                                            Used for backpropagation updates. Default is None.

        Methods:
            attach(layer): Attach the neurons from this layer to another layer, needed for backpropagation.
            init_layer(num_input): Initialize the weights of each neuron in the layer.
                                   Receives the number of inputs to determine the number of weights.
            predict(row): Calculate the activations for the full layer given a row of data as input.

        Note:
            The activation function is specified during the creation of the Layer object and can be one of:
            'sigmoid', 'relu', 'tanh', or 'softmax'.

            For binary classification, use 'sigmoid' for the output layer activation.
            For multiclass classification, use 'softmax' for the output layer activation.
        """
    def __init__(self, num_neuron, activation='sigmoid', is_output_layer=False, weight_decay=0.01, position_in_layer=None):
        """
                Constructor for Layer class.

                Parameters:
                    num_neuron (int): The number of neurons in the layer.
                    activation (str, optional): The activation function to be used for the neurons in the layer.
                                                Default is 'sigmoid'.
                    is_output_layer (bool, optional): Indicates if the layer is the output layer. Default is False.
                    weight_decay (float, optional): The weight decay parameter to apply to the neurons in the layer.
                                                    Default is 0.01.
                    position_in_layer (int or None, optional): The position index of the layer in the network architecture.
                                                               Used for backpropagation updates. Default is None.
        """
        # Will create that much neurons in this layer
        self.is_output_layer = is_output_layer
        self.neurons = []
        self.weight_decay = weight_decay

        # This is used for the backpropagation update
        self.updated_weights = []
        # This is used for the backpropagation update
        self.position_in_layer = position_in_layer

        # Activation function
        self.output = activation

        for i in range(num_neuron):
            # Create neuron
            neuron = Neuron(i, is_output_neuron=is_output_layer)
            self.neurons.append(neuron)

    def attach(self, layer):
        """
                Attach the neurons from this layer to another layer.
                This function is needed for backpropagation updates.

                Parameters:
                    layer (Layer): The next layer to which the neurons from this layer will be attached.
        """
        # Iterate over the neurons in the current layer and attach
        # them to the next layer
        for in_neuron in self.neurons:
            in_neuron.attach_to_output(layer.neurons)

    def init_layer(self, num_input):
        """
                Initialize the weights of each neuron in the layer.
                This function sets up the weights when the number of inputs for a neuron is known.

                Parameters:
                    num_input (int): The number of input features for each neuron in the layer.
        """
        # Iterate over each of the neuron and initialize
        # the weights that connect with the previous layer
        for neuron in self.neurons:
            neuron.init_weights(num_input)
            neuron.weight_decay = self.weight_decay

    def predict(self, row):
        """
                Calculate the activations for the full layer given a row of data as input.

                Parameters:
                    row (numpy.ndarray): A row of data as input to the layer.

                Returns:
                    list: A list containing the activations for each neuron in the layer.
        """
        bias = np.array([1])  # Create a bias term with value 1
        row_with_bias = np.concatenate([row, bias])  # Add the bias term to the input row

        activations = [neuron.predict(row_with_bias, self.output) for neuron in self.neurons]
        return activations

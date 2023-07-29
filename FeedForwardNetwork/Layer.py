from FeedForwardNetwork.Neuron import Neuron


class Layer():
    '''
        Layer is modelizing a layer in the fully-connected-feedforward neural network architecture.
        It will play the role of connecting everything together inside and will be doing the backpropagation 
        update.
    '''

    def __init__(self, num_neuron, is_output_layer=False, weight_decay=0.01):
        # Will create that much neurons in this layer
        self.is_output_layer = is_output_layer
        self.neurons = []
        self.weight_decay = weight_decay

        for i in range(num_neuron):
            # Create neuron
            neuron = Neuron(i, is_output_neuron=is_output_layer)
            self.neurons.append(neuron)

    def attach(self, layer):
        '''
            This function attach the neurons from this layer to another one
            This is needed for the backpropagation algorithm
        '''
        # Iterate over the neurons in the current layer and attach
        # them to the next layer
        for in_neuron in self.neurons:
            in_neuron.attach_to_output(layer.neurons)

    def init_layer(self, num_input):
        '''
            This will initialize the weights of each neuron in the layer.
            By giving the right num_input it will spawn the right number of weights.
            Additionally, it will pass the weight decay parameter to each neuron.
        '''

        # Iterate over each of the neuron and initialize
        # the weights that connect with the previous layer
        for neuron in self.neurons:
            neuron.init_weights(num_input)
            neuron.weight_decay = self.weight_decay

    def predict(self, row):
        '''
            This will calculate the activations for the full layer given the row of data
            streaming in.
        '''
        row.append(1)  # need to add the bias
        activations = [neuron.predict(row) for neuron in self.neurons]
        return activations

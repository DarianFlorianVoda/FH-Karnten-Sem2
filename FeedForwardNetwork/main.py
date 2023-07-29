import numpy as np

from FeedForwardNetwork.MLP import MultiLayerPerceptron
from FeedForwardNetwork.half_moon import create_half_moon, label_half_moon


def XOR_example():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    # Init the parameters for the network
    clf = MultiLayerPerceptron(learning_rate=0.1, num_iteration=100000)

    # Create the architecture backward
    clf.add_output_layer(num_neuron=1)
    clf.add_hidden_layer(num_neuron=3)
    clf.add_hidden_layer(num_neuron=2)

    # Train the network
    print(clf.fit(X, y))


def halfmoon_linear_example():
    X, y = label_half_moon(n=2000, w=0.2, r=0.6, d=-0.1)

    # Init the parameters for the network
    clf = MultiLayerPerceptron(learning_rate=0.1, num_iteration=100000)

    # Create the architecture backward
    clf.add_output_layer(num_neuron=1)
    clf.add_hidden_layer(num_neuron=3)
    clf.add_hidden_layer(num_neuron=2)

    # Train the network
    print(clf.fit(X, y))


# x1, y1, x2, y2 = create_half_moon(2000, 0.2, 0.6, 0.1)

# X, y = label_half_moon()

# print(X)
# print(y)

halfmoon_linear_example()
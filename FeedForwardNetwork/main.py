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
    clf.add_output_layer(num_neuron=2)
    clf.add_hidden_layer(num_neuron=3)

    # Train the network
    print(clf.fit(X, y))


def halfmoon_nonlinear_example():
    X, y = label_half_moon(n=2000, w=0.2, r=0.6, d=0.2)

    # Init the parameters for the network
    clf = MultiLayerPerceptron(learning_rate=0.1, num_iteration=100000, weight_decay=0.001)

    # Create the architecture backward
    clf.add_output_layer(num_neuron=2)
    clf.add_hidden_layer(num_neuron=3)
    clf.add_hidden_layer(num_neuron=2)

    # Train the network
    print(clf.fit(X, y))
def load_data_from_files(file_paths):
    data = []
    labels = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = list(map(float, line.strip().split()))
                data.append(values[:-1])  # All values except the last one are features
                labels.append(values[-1])  # Last value is the label

    return np.array(data), np.array(labels)

def train_from_files(file_paths):
    X, y = load_data_from_files(file_paths)

    # Init the parameters for the network
    clf = MultiLayerPerceptron(learning_rate=0.1, num_iteration=100000, weight_decay=0.01)

    # Create the architecture backward
    clf.add_output_layer(num_neuron=4)
    clf.add_hidden_layer(num_neuron=2)
    clf.add_hidden_layer(num_neuron=4)

    # Train the network
    clf.fit(X, y)


# x1, y1, x2, y2 = create_half_moon(2000, 0.2, 0.6, 0.1)

# X, y = label_half_moon()

# print(X)
# print(y)
# halfmoon_linear_example()

halfmoon_nonlinear_example()


# Provide the paths to the four ASCII files
file_paths = ['C:\\Users\\daria\\PycharmProjects\\FH-Karnten-Sem2\\FeedForwardNetwork\\NoisySymbols\\10Times10NoisyCycle.txt',
              'C:\\Users\\daria\\PycharmProjects\\FH-Karnten-Sem2\\FeedForwardNetwork\\NoisySymbols\\10Times10NoisyMinus.txt',
              'C:\\Users\\daria\\PycharmProjects\\FH-Karnten-Sem2\\FeedForwardNetwork\\NoisySymbols\\10Times10NoisyPlus.txt',
              'C:\\Users\\daria\\PycharmProjects\\FH-Karnten-Sem2\\FeedForwardNetwork\\NoisySymbols\\10Times10NoisyX.txt']

# train_from_files(file_paths)
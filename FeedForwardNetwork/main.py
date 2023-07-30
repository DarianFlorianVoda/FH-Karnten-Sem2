import numpy as np
from sklearn.model_selection import train_test_split

from FeedForwardNetwork.MLP import MultiLayerPerceptron
from FeedForwardNetwork.half_moon import create_half_moon, label_half_moon


def XOR_example():
    """
        Example of using MultiLayerPerceptron for XOR classification.

        XOR is a classic binary classification problem where the output is 1 if the input
        features are different and 0 if they are the same.

        This function initializes the parameters for the network, creates the architecture
        backward with one output layer and two hidden layers, and trains the network using
        stochastic gradient descent.

        The output is the mean error after training.

        Note: The MultiLayerPerceptron class and other required functions should be imported
        or defined in the same file to use this function.

        Returns:
            float: The mean error after training the network.
    """
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    # Init the parameters for the network
    clf = MultiLayerPerceptron(learning_rate=0.1, num_iteration=10000)

    # Create the architecture backward
    clf.add_output_layer(num_neurons=1)
    clf.add_hidden_layer(num_neuron=3)
    clf.add_hidden_layer(num_neuron=2)

    # Train the network
    print(clf.fit(X, y))


def halfmoon_linear_example():
    """
        Example of using MultiLayerPerceptron for linearly separable half-moon classification.

        This function generates a dataset of linearly separable half-moon shapes, splits it
        into training and testing sets, initializes the parameters for the network, creates
        the architecture backward with one output layer using linear activation, and one
        hidden layer using the hyperbolic tangent (tanh) activation function. The function
        then trains the network using stochastic gradient descent and evaluates its accuracy
        on the test set.

        The output is the accuracy of the trained network on the test set.

        Note: The MultiLayerPerceptron class and other required functions should be imported
        or defined in the same file to use this function.

        Returns:
            float: The accuracy of the trained network on the test set.
    """
    X, y = label_half_moon(n=2000, w=0.2, r=0.6, d=-0.1)

    y = np.array(y)  # Convert y to a NumPy array

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Init the parameters for the network
    clf = MultiLayerPerceptron(learning_rate=0.01, num_iteration=10000, weight_decay=0.01)

    # Create the architecture backward
    clf.add_output_layer(num_neurons=1)  # Use sigmoid activation for the output layer
    clf.add_hidden_layer(num_neuron=10, activation='tanh')  # Use tanh activation for the hidden layer

    # Train the network
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = []
    for row in X_test:
        y_pred.append(np.round(clf.predict(row)))  # Convert probabilities to binary predictions (0 or 1)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy}")


def halfmoon_nonlinear_example():
    """
        Example of using MultiLayerPerceptron for nonlinearly separable half-moon classification.

        This function generates a dataset of nonlinearly separable half-moon shapes, splits it
        into training and testing sets, initializes the parameters for the network, creates
        the architecture backward with one output layer, one hidden layer using the hyperbolic
        tangent (tanh) activation function, and another hidden layer using the rectified linear
        unit (ReLU) activation function. The function then trains the network using stochastic
        gradient descent and evaluates its accuracy on the test set.

        The output is the accuracy of the trained network on the test set.

        Note: The MultiLayerPerceptron class and other required functions should be imported
        or defined in the same file to use this function.

        Returns:
            float: The accuracy of the trained network on the test set.
    """
    X, y = label_half_moon(n=2000, w=0.2, r=0.6, d=0.2)

    y = np.array(y)  # Convert y to a NumPy array

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Init the parameters for the network (using default hyperparameters)
    clf = MultiLayerPerceptron(learning_rate=0.1, num_iteration=100000, weight_decay=0.01)

    # Create the architecture backward
    clf.add_output_layer(num_neurons=1)
    clf.add_hidden_layer(num_neuron=3, activation='tanh')
    clf.add_hidden_layer(num_neuron=2, activation='relu')

    # Train the network
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = []
    for row in X_test:
        y_pred.append(np.argmax(clf.predict(row)))  # Convert probabilities to binary predictions (0 or 1)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy}")


def load_data_from_files(file_paths):
    """
        Load data and labels from text files.

        This function reads data and labels from text files specified by their file paths.
        Each line in the files represents a data point, where the features are separated
        by spaces, and the last value on each line is the label. The data and labels are
        then returned as NumPy arrays.

        Parameters:
            file_paths (list): A list of file paths to load data from.

        Returns:
            numpy.ndarray, numpy.ndarray: The data and labels loaded from the files.
    """
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
    """
        Example of using MultiLayerPerceptron to train a network using data from files.

        This function loads data and labels from text files specified by their file paths,
        converts the labels to binary (0 or 1), splits the data into training and testing sets,
        initializes the parameters for the network, creates the architecture backward with
        one output layer and two hidden layers, and trains the network using stochastic
        gradient descent. Finally, it prints the prediction error rate from a test set.

        Note: The MultiLayerPerceptron class and other required functions should be imported
        or defined in the same file to use this function.

        Parameters:
            file_paths (list): A list of file paths to load data from.

        Returns:
            None
    """
    # Load data from files
    X, y = load_data_from_files(file_paths)

    # Convert labels to binary (0 or 1)
    y = (y + 1) // 2

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Init the parameters for the network
    clf = MultiLayerPerceptron(learning_rate=0.1, num_iteration=10000, weight_decay=0.01)

    # Create the architecture backward
    clf.add_output_layer(num_neurons=4)
    clf.add_hidden_layer(num_neuron=2)
    clf.add_hidden_layer(num_neuron=4)

    # Train the network
    clf.fit(X_train, y_train)

    print("Prediction error rate from a test set:")
    print(clf.predict(X_test[0]))

# Provide the paths to the four ASCII files
file_paths = [
    'C:\\Users\\daria\\PycharmProjects\\FH-Karnten-Sem2\\FeedForwardNetwork\\NoisySymbols\\10Times10NoisyCycle.txt',
    'C:\\Users\\daria\\PycharmProjects\\FH-Karnten-Sem2\\FeedForwardNetwork\\NoisySymbols\\10Times10NoisyMinus.txt',
    'C:\\Users\\daria\\PycharmProjects\\FH-Karnten-Sem2\\FeedForwardNetwork\\NoisySymbols\\10Times10NoisyPlus.txt',
    'C:\\Users\\daria\\PycharmProjects\\FH-Karnten-Sem2\\FeedForwardNetwork\\NoisySymbols\\10Times10NoisyX.txt']

######## EXAMPLES

# x1, y1, x2, y2 = create_half_moon(2000, 0.2, 0.6, 0.1)

# X, y = label_half_moon()

# print(X)
# print(y)

# XOR_example()

halfmoon_linear_example()

# halfmoon_nonlinear_example()

# train_from_files(file_paths)

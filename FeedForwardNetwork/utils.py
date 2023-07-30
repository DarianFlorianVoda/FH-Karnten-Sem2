import numpy as np


# Activation Functions
def tanh(x):
    """Hyperbolic tangent activation function.

    Parameters:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying the hyperbolic tangent activation element-wise."""
    return np.tanh(x)


def d_tanh(x):
    """
       Derivative of the hyperbolic tangent activation function.

       Parameters:
           x (numpy.ndarray): Input array.

       Returns:
           numpy.ndarray: Derivative of the hyperbolic tangent activation function with respect to input x.
       """
    return 1 - np.square(np.tanh(x))


def sigmoid(x):
    """
        Sigmoid activation function.

        Parameters:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying the sigmoid activation element-wise.
        """
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    """
        Derivative of the sigmoid activation function.

        Parameters:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Derivative of the sigmoid activation function with respect to input x.
    """
    return (1 - sigmoid(x)) * sigmoid(x)


def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying the ReLU activation element-wise.
    """
    return np.maximum(0, x)


def softmax(x):
    """
        Softmax activation function for multiclass classification.

        Parameters:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying the softmax activation along the specified axis.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

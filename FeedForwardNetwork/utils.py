import numpy as np


# Activation Functions
def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.square(np.tanh(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)


# Loss Functions
def logloss(y, a):
    return -(y * np.log(a) + (1 - y) * np.log(1 - a))


def softmax(self, x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def d_logloss(y, a):
    return (a - y) / (a * (1 - a))

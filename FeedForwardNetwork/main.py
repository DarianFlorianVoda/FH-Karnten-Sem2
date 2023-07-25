import numpy as np

from FeedForwardNetwork.Layer import Layer
from FeedForwardNetwork.utils import logloss, d_logloss
import matplotlib.pyplot as plt

x_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # dim x m
y_train = np.array([[0, 1, 1, 0]]) # 1 x m

m = 4
epochs = 1500

layers = [Layer(2, 3, 'tanh'), Layer(3, 1, 'sigmoid')]
costs = [] # to plot graph

for epoch in range(epochs):
    # Feedforward
    A = x_train
    for layer in layers:
        A = layer.feedforward(A)

    # Calulate cost to plot graph
    cost = 1/m * np.sum(logloss(y_train, A))
    costs.append(cost)

    # Backpropagation
    dA = d_logloss(y_train, A)
    for layer in reversed(layers):
        dA = layer.backprop(dA)


# Making predictions
A = x_train
for layer in layers:
    A = layer.feedforward(A)
print(A)


plt.plot(range(epochs), costs)
plt.show()
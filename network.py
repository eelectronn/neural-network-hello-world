import numpy as np


class Network:
    def __init__(self, layers):
        self.size = layers
        self.bias = np.asarray([[np.random.randn() for j in range(layers[i])] for i in range(1, len(layers))])
        self.weight = np.asarray([[[np.random.randn() for k in range(layers[i-1])] for j in range(layers[i])] for i in range(1, len(layers))])

    def forward_feed(self, activation):
        for weight_set, bias_set in zip(self.weight, self.bias):
            activation = sigmoid(np.matmul(weight_set, activation) + bias_set)
        return activation


def sigmoid(num):
    return 1.0/(1.0+np.exp(-num))

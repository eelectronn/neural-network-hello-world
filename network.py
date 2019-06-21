import numpy as np


class Network:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.size = layers
        self.bias = np.asarray([[np.random.randn() for _ in range(layers[i])] for i in range(1, len(layers))])
        self.weight = np.asarray([[[np.random.randn() for _ in range(layers[i-1])] for _ in range(layers[i])] for i in range(1, len(layers))])

    def forward_feed(self, activation):
        for weight_set, bias_set in zip(self.weight, self.bias):
            activation = sigmoid(np.matmul(weight_set, activation) + bias_set)
        return activation

    def get_gradient(self, data, expected):
        activations = [data]
        zs = [data]
        entry = data
        for weight_set, bias_set in zip(self.weight, self.bias):
            entry = np.matmul(weight_set, entry) + bias_set
            activation = sigmoid(entry)
            zs.append(entry)
            activations.append(activation)
        delta = 2*(activations[-1] - expected)
        delta_weight = [np.zeros(w.shape) for w in self.weight]
        delta_bias = [np.zeros(b.shape) for b in self.bias]
        for i in range(self.num_layers-1):
            z_prime = sigmoid_prime(zs[i-1])
            delta_bias[-i-1] = delta*sigmoid_prime(z_prime)
            delta_weight[-i-1] = np.matmul(np.asmatrix(delta_bias[-i-1]).T, np.asmatrix(activations[i-2]))
            delta = np.matmul(np.asmatrix(delta*z_prime), np.asmatrix(self.weight[-i-1]))
        return delta_weight, delta_bias


def sigmoid(num):
    return 1.0/(1.0+np.exp(-num))


def sigmoid_prime(num):
    return sigmoid(num)*sigmoid(num)*np.exp(-num)

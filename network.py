import numpy as np


class Network:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.size = layers
        self.bias = np.asarray([np.asarray([np.random.randn() for _ in range(layers[i])]) for i in range(1, len(layers))])
        self.weight = np.asarray([np.asarray([np.asarray([np.random.randn() for _ in range(layers[i-1])]) for _ in range(layers[i])]) for i in range(1, len(layers))])

    def forward_feed(self, activation):
        for weight_set, bias_set in zip(self.weight, self.bias):
            activation = sigmoid(np.matmul(weight_set, activation) + bias_set)
        return activation

    def get_gradient(self, data, expected):
        activations, zs = [data], [data]
        for weight_set, bias_set in zip(self.weight, self.bias):
            zs.append(np.matmul(weight_set, activations[-1]) + bias_set)
            activations.append(sigmoid(zs[-1]))
        delta = 2*(activations[-1] - expected)
        delta_w = np.asarray([np.zeros(w.shape) for w in self.weight])
        delta_b = np.asarray([np.zeros(b.shape) for b in self.bias])
        for i in range(self.num_layers-1):
            z_prime = np.asarray(sigmoid_prime(zs[-i-1]))
            delta_b[-i-1] = delta*z_prime
            delta_w[-i-1] = np.asarray(np.matmul(np.asmatrix(delta_b[-i-1]).T, np.asmatrix(activations[i-2])))
            delta = np.asarray(np.matmul(np.asmatrix(delta*z_prime), np.asmatrix(self.weight[-i-1])))
        return delta_w, delta_b

    def learn(self, batch, rate):
        avg_delta_w = np.asarray([np.zeros(w.shape) for w in self.weight])
        avg_delta_b = np.asarray([np.zeros(b.shape) for b in self.bias])
        for data, expected in batch:
            delta_w, delta_b = self.get_gradient(data, expected)
            avg_delta_w = avg_delta_b + delta_w
            avg_delta_b = avg_delta_b + delta_b
        self.weight = self.weight - (rate*avg_delta_w/len(batch))
        self.bias = self.bias - (rate*avg_delta_b/len(batch))


def sigmoid(num):
    return 1.0/(1.0+np.exp(-num))


def sigmoid_prime(num):
    return sigmoid(num)*sigmoid(num)*np.exp(-num)

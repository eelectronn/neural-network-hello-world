import numpy as np


class Network:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.size = layers
        self.bias = np.asarray([np.asarray([np.random.randn() for _ in range(layers[i])]) for i in range(1, len(layers))])
        self.weight = np.asarray([np.asarray([np.asarray([np.random.randn() for _ in range(layers[i-1])]) for _ in range(layers[i])]) for i in range(1, len(layers))])

    def forward_feed(self, activation):
        activation = sigmoid(activation)
        for weight_set, bias_set in zip(self.weight, self.bias):
            activation = sigmoid(np.matmul(weight_set, activation) + bias_set)
        return activation

    def get_gradient(self, data, expected):
        activations, zs = [sigmoid(data)], [sigmoid(data)]
        for weight_set, bias_set in zip(self.weight, self.bias):
            zs.append(np.matmul(weight_set, activations[-1]) + bias_set)
            activations.append(sigmoid(zs[-1]))
        delta = 2*(activations[-1] - expected)
        delta_w = np.asarray([np.zeros(w.shape) for w in self.weight])
        delta_b = np.asarray([np.zeros(b.shape) for b in self.bias])
        for i in range(self.num_layers-1):
            z_prime = np.asarray(sigmoid_prime(zs[-i-1]))
            delta_b[-i-1] = delta*z_prime
            delta_w[-i-1] = np.asarray(np.matmul(np.asmatrix(delta_b[-i-1]).T, np.asmatrix(activations[-i-2])))
            delta = np.asarray(np.matmul(delta_b[-i-1], self.weight[-i-1]))
        return delta_w, delta_b

    def learn(self, batch_data, batch_expected, rate):
        avg_delta_w = np.asarray([np.zeros(w.shape) for w in self.weight])
        avg_delta_b = np.asarray([np.zeros(b.shape) for b in self.bias])
        for data, expected in zip(batch_data, batch_expected):
            delta_w, delta_b = self.get_gradient(data, expected)
            avg_delta_w = avg_delta_w + delta_w
            avg_delta_b = avg_delta_b + delta_b
        self.weight = self.weight - (rate*avg_delta_w/len(batch_data))
        for b, delta_b in zip(self.bias, avg_delta_b):
            b = b - (rate/len(batch_data)*delta_b)

    def stochastic_gradient_descent(self, training_data, rate, batch_size, iteration):
        for i in range(iteration):
            state = np.random.get_state()
            np.random.shuffle(training_data[0])
            np.random.set_state(state)
            np.random.shuffle(training_data[1])
            data_batches = [training_data[0][j:j+batch_size] for j in range(0, len(training_data[0]), batch_size)]
            label_batches = [training_data[1][j:j+batch_size] for j in range(0, len(training_data[1]), batch_size)]
            for batch_data, batch_label in zip(data_batches, label_batches):
                self.learn(batch_data, batch_label, rate)
            print('Iteration ' + str(i) + ' completed.')

    def test(self, testing_data):
        data = testing_data[0]
        label = testing_data[1]
        total_correct = 0
        for d, l in zip(data, label):
            output_array = self.forward_feed(d)
            network_answer = 0
            for i in range(len(output_array)):
                if output_array[i] > output_array[network_answer]:
                    network_answer = i
            if l == network_answer:
                total_correct += 1
        return total_correct/len(data)


def sigmoid(num):
    return 1.0/(1.0+np.exp(-num))


def sigmoid_prime(num):
    return sigmoid(num)*sigmoid(num)*np.exp(-num)

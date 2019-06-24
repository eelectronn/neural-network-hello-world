import network as nw
import data_loader as dl

training_data, testing_data = dl.load_training_data()
print('data loaded')
net = nw.Network([784, 16, 16, 10])
net.stochastic_gradient_descent(training_data, 0.5, 10, 20)
rate = net.test(testing_data)
print(rate)
net.stochastic_gradient_descent(training_data, 0.5, 10, 20)
rate = net.test(testing_data)
print(rate)
net.stochastic_gradient_descent(training_data, 0.5, 10, 20)
rate = net.test(testing_data)
print(rate)

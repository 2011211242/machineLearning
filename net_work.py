#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import input_data
import mnist_loader

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    sigm = sigmoid(z)
    return sigm * (1 - sigm)

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost from associated with output "a" """
        return np.sum(np.nan_to_num(-y * np.log(a) - 
            (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer"""
        return (a - y)

class NetWork(object):
    def __init__(self, shapes=[784, 30, 10], cost=CrossEntropyCost):
        """Define the network using "shapes" """
        self.num_layers = len(shapes)
        self.shapes = shapes
        self.biases = [np.random.randn(y, 1) for y in self.shapes[1:]]
        self.weights = [np.random.randn(y, x) 
                for x, y in zip(self.shapes[:-1], self.shapes[1:])]
        print(self.weights[0].shape)
        print(self.weights[1].shape)
        self.cost = cost

    def feedforward(self, a):
        """Return output of the network if "a" is input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        #print(type([int(x == y) for (x, y) in results]))
        return sum([int(x == y) for (x, y) in results]) / len(results)

    def backprop(self, x, y):
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        delta = (self.cost).delta(zs[-1], activations[-1], y)

        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp

            nable_b[-l] = delta
            nable_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nable_b, nable_w)

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nable_b = [np.zeros(b.shape) for b in self.biases]
        nable_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nable_b, delta_nable_w = self.backprop(x, y)
            nable_b = [nb + dnb for nb, dnb in zip(nable_b, delta_nable_b)]
            #print(len(delta_nable_w))
            #print(delta_nable_w[0].shape)
            #print(delta_nable_w[1].shape)

            #print(nable_w[0].shape)
            #print(nable_w[1].shape)
            nable_w = [nw + dnw for nw, dnw in zip(nable_w, delta_nable_w)]

        self.weights = [(1 - eta * (lmbda / n)) * w - (eta/len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nable_w)] 

        self.biases = [b - (eta/len(mini_batch)) * nb
                        for b, nb in zip(self.biases, nable_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0, evaluation_data=None):
        n = len(training_data)
        for j in xrange(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] \
                    for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            accuracy = self.accuracy(evaluation_data, convert=True)
            print("Epochs %d, accuracy %f" %(j, accuracy))



    
    """
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nable_b = 
    """

    
if __name__ == "__main__":
    network = NetWork()
    training_data, validation_data, test_data = input_data.read_data_sets("MNIST_data", one_hot = True)
    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(len(training_data))
    network.SGD(training_data, 100, 30, 0.5, 1e-2, validation_data)


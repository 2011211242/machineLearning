#encoding=utf-8
import numpy as np

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
    def delta(a, y, z):
        """Return the error delta from the output layer"""
        return (a - y)

class NetWork(object):
    def __init__(self, shapes=[784, 10], cost=CrossEntropyCost):
        """Define the network using "shapes" """
        self.num_layers = len(shapes)
        self.shapes = shapes
        self.biases = [np.random.randn(y, 1) for y in shapes[1:]]
        self.weights = [np.random.randn(y, x) 
                for x, y in zip(shapes[:-1], shapes[1:])]

    def feedforward(self, a):
        """Return output of the network if "a" is input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        nable_w = [np.zeros(w.shape) for w in self.weights]
        nable_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        delta = (self.cost).delta(activations[-1], y, z[-1])

        nable_b[-1] = delta
        nable_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]


if __name__ == "__main__":
    NetWork()
    

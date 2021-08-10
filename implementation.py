import random
import numpy as np
"""
This module contains code for mini_batch gradient descent
for a feedforward NN. And we are calculating gradients using
backpropogation .
"""

def sigmoid(z):
    """
    Returning sigmoid of a given vector
    """
    return 1.0/(1.0 + np.exp(-z))
def derivative_sigmoid(z):
    """
    Returns derivative of sigmoid(z)
    """
    return sigmoid(z)*(1-sigmoid(z))


class Model(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes=sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """Return the output with given weights and biases if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def update_mini_batch(self, mini_batch, eta):
        """Update weights and biases t using backpropagation to a
        single mini batch and``eta``
        is the learning rate."""
        temp_b = [np.zeros(b.shape) for b in self.biases]
        temp_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_temp_b, delta_temp_w = self.backprop(x, y)
            temp_b = [nb+dnb for nb, dnb in zip(temp_b, delta_temp_b)]
            temp_w = [nw+dnw for nw, dnw in zip(temp_w, delta_temp_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, temp_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, temp_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data):
        """Train the neural network using mini-batch stochastic
        gradient descent.
        """
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))


    def backprop(self, x, y):
        """Return a tuple ``(temp_b, temp_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        temp_b = [np.zeros(b.shape) for b in self.biases]
        temp_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            derivative_sigmoid(zs[-1])
        temp_b[-1] = delta
        temp_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = derivative_sigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            temp_b[-l] = delta
            temp_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (temp_b, temp_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

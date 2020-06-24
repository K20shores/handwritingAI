import numpy as np
from tqdm import tqdm
import random
random.seed(0)
import sys

class NeuralHandwritingNet:
    def __init__(self, config):
        self.num_layers = len(config.layers)
        self.layers = config.layers
        self.biases = config.biases
        self.weights = config.weights
        self.eta = config.eta
        self.mini_batch_size = config.mini_batch_size
        self.epochs = config.epochs

    def evaluate(self, test_data, progress_bar = True):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []
        if progress_bar:
            iterable = tqdm(test_data, desc='Predicting', file=sys.stdout, dynamic_ncols=True) 
        else:
            iterable = test_data
        for x, y in iterable:
            result = np.argmax(self.__feedforward(x))
            label = np.argmax(y)
            test_results.append((result, label))
        results = [x == y for (x, y) in test_results]
        return sum(1 for x in results if x), results

    def SGD(self, training_data, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent.  

        The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """

        if test_data: n_test = len(test_data)
        n = len(training_data)
        progress_bar = tqdm(range(self.epochs), desc='Training', file=sys.stdout, dynamic_ncols=True)
        for j in progress_bar:
            random.shuffle(training_data)
            mini_batches = np.array_split(training_data, np.floor(n / self.mini_batch_size))
            for mini_batch in mini_batches:
                self.__update_mini_batch(mini_batch)
            if test_data:
                success = 100 * self.evaluate(test_data, progress_bar = False)[0] / n_test
                progress_bar.set_description(f'Training (success rate {success:.2f}%)')

    def __update_mini_batch(self, mini_batch):
        """Brackpopogate to update weights and biases using gradient descent

        Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.__backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(self.eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def __feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.__sigmoid(np.dot(w, a) + b)
        return a

    def __backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.__sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.__cost_derivative(activations[-1], y) * \
            self.__sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sig_derivative = self.__sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sig_derivative
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return (nabla_b, nabla_w)
    
    def __cost_derivative(self, y_hat, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return y_hat - y

    def __sigmoid(self, x):
        #could also use https://stackoverflow.com/a/29863846/5217293
        return 1. / (1. + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))


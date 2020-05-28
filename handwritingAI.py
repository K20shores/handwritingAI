import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import argparse
import json, codecs
from handwriting_utils.data_reader import LabelDataReader
from handwriting_utils.data_reader import ImageDataReader

class NeuralHandwritingNetConfig:
    def __init__(self):
        self.sizes = None
        self.biases = None
        self.weights = None
        self.mini_batch_size = None
        self.epochs = None
        self.eta = None

    def read_config(self, file_path):
        json_text = codecs.open(file_path, 'r', encoding='utf-8').read()
        json_data = json.loads(json_text)
        self.sizes = json_data['sizes']
        self.biases = np.array(json_data['biases'])
        self.weights = np.array(json_data['weights'])
        self.mini_batch_size = json_data['mini_batch_size']
        self.epochs = json_data['epochs']
        self.eta = json_data['eta']

    def save_config(self, file_path):
        sizes = self.sizes
        biases = [x.tolist() for x in self.biases]
        weights = [x.tolist() for x in self.weights]
        config = {
                'sizes':sizes,
                'biases': biases,
                'weights': weights,
                'mini_batch_size': self.mini_batch_size,
                'epochs': self.epochs,
                'eta': self.eta
            }
        json.dump(config,
                codecs.open(file_path, 'w', encoding='utf-8'),
                separators=(',', ':'), sort_keys=True, indent=4)

class NeuralHandwritingNet:
    def __init__(self, config):
        self.num_layers = len(config.sizes)
        self.sizes = config.sizes
        self.biases = config.biases
        self.weights = config.weights
        self.eta = config.eta
        self.mini_batch_size = config.mini_batch_size
        self.epochs = config.epochs

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.__feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

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
        #for j in tqdm(range(self.epochs), desc='Training batches'):
        for j in range(self.epochs):
            random.shuffle(training_data)
            mini_batches = np.array_split(training_data, np.floor(n / self.mini_batch_size))
            for self.mini_batch in mini_batches:
                self.__update_mini_batch()
            if test_data:
                print(f"Epoch {j}: {100*self.evaluate(test_data) / n_test}%")
            else:
                print(f"Epoch {j} complete")

    def __update_mini_batch(self):
        """Brackpopogate to update weights and biases using gradient descent

        Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in self.mini_batch:
            delta_nabla_b, delta_nabla_w = self.__backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(self.eta/len(self.mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.eta/len(self.mini_batch))*nb 
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

def show_image(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_image_grid(data):
    n_rows = int(np.ceil(np.sqrt(data.shape[0])))
    n_cols = n_rows
    fig, axs = plt.subplots(n_rows, n_cols)

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axs[row][col]
            idx = n_cols * row + col
            if (idx >= data.shape[0]):
                ax.set_visible(False)
            else:
                ax.imshow(data[idx], cmap='binary')
                ax.set_xticks([])
                ax.set_yticks([])
    plt.show()

def get_training_data(n, test_data = False):
    n = int(n) if test_data else None
    label_reader = LabelDataReader("training_data/train-labels-idx1-ubyte")
    labels = label_reader.read(n)

    image_reader = ImageDataReader("training_data/train-images-idx3-ubyte")
    images = image_reader.read(n)
    image_size = image_reader.image_size

    test_data = None
    if test_data:
        test_data_length = n // 5
        test_data_labels = labels[-test_data_length:]
        labels = labels[:-test_data_length]
        test_data_images = images[-test_data_length:]
        images = images[:-test_data_length:]
        test_data = [x for x in zip(test_data_images, test_data_labels)]

    training_data = [x for x in zip(images, labels)]

    return training_data, test_data

def get_test_data():
    label_reader = LabelDataReader("test_data/t10k-labels-idx1-ubyte")
    labels = label_reader.read()

    image_reader = ImageDataReader("test_data/t10k-images-idx3-ubyte")
    images = image_reader.read()

    return [x for x in zip(images, labels)]

def make_config(sizes, eta, mini_batch_size, epochs):
    config = NeuralHandwritingNetConfig()
    config.sizes = sizes
    config.eta = eta
    config.mini_batch_size = mini_batch_size
    config.epochs = epochs
    config.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    config.weights = [np.random.randn(y, x)
                    for x, y in zip(sizes[:-1], sizes[1:])]
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure the handwriting AI')
    parser.add_argument('-E', '--Epochs', type=int, help='Number of epochs', default=30)
    parser.add_argument('-m', '--minibatchsize', type=int, help='Size of the minibatches', default=10)
    parser.add_argument('-e', '--eta', type=int, help='Eta, the learning rate', default=3)
    parser.add_argument('--hidden_layers', metavar='N', type=int, help='A list of integers representing the number of nodes in each hidden layers.', nargs='+', default=[100])
    parser.add_argument('-c', '--config', type=str, help='The config file to use to initialize the AI or save the result of a training run', default=None)
    parser.add_argument('--train', action='store_true', help='Use this to elect to train the AI', dest='train')
    parser.add_argument('--no-train', action='store_false', help='Use this to elect to use the AI to predict values', dest='train')
    parser.set_defaults(train=True)
    args = parser.parse_args()

    train = args.train
    eta = args.eta
    mini_batch_size = args.minibatchsize
    epochs = args.Epochs
    layers = args.hidden_layers
    config_file = args.config

    if config_file is None:
        raise ValueError('config cannot be None')

    if train:
        # input layers
        layers.insert(0, 28 * 28)
        # output layers
        layers.append(10)
        config = make_config(sizes = layers, 
                eta = eta, mini_batch_size = mini_batch_size, epochs = epochs)
    else:
        config = NeuralHandwritingNetConfig()
        config.read_config(config_file)

    ai = NeuralHandwritingNet(config)

    if train:
        training, test = get_training_data(6e4, test_data = True)
        ai.SGD(training, test_data=test)
        config.weights = ai.weights
        config.biases = ai.biases
        config.save_config(config_file)
    else:
        test_data = get_test_data()
        ncorrect = ai.evaluate(test_data)
        print(f"Correctly predicted test data: {100 * ncorrect / len(test_data)}%")


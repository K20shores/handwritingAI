import numpy as np
import argparse

from mnist.ai import NeuralHandwritingNet
from mnist.utils import config as cfg
from mnist.utils.image_viewer import *
from mnist import data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configure the handwriting AI')
    parser.add_argument('-E', '--Epochs', type=int, help='Number of epochs', default=30)
    parser.add_argument('-m', '--minibatchsize', type=int, help='Size of the minibatches', default=10)
    parser.add_argument('-e', '--eta', type=int, help='Eta, the learning rate', default=3)
    parser.add_argument('--hidden_layers', metavar='N', type=int, help='A list of integers representing the number of nodes in each hidden layers.', nargs='+', default=[100])
    parser.add_argument('-c', '--config', type=str, help='The config file to use to initialize the AI or save the result of a training run', default=None)
    parser.add_argument('--train', action='store_true', help='Use this to elect to train the AI', dest='train')
    parser.add_argument('--predict', action='store_false', help='Use this to elect to use the AI to predict values', dest='train')
    parser.add_argument('-s', '--sumarize-config', help='Use this to print out stats abot the config', action='store_true', dest='summarize_config')
    parser.set_defaults(train=None)
    args = parser.parse_args()

    train = args.train
    eta = args.eta
    mini_batch_size = args.minibatchsize
    epochs = args.Epochs
    layers = args.hidden_layers
    config_file = args.config

    if config_file is None:
        raise ValueError('config cannot be None')

    config = None
    if train is not None:
        if train:
            # input layers
            layers.insert(0, 28 * 28)
            # output layers
            layers.append(10)
            config = cfg.make_config(layers = layers, 
                    eta = eta, mini_batch_size = mini_batch_size, epochs = epochs)
        else:
            config = cfg.from_file(config_file)

        ai = NeuralHandwritingNet(config)

        if train:
            training, test = data.get_training_data()
            ai.SGD(training, test_data=test)
            config.weights = ai.weights
            config.biases = ai.biases
            config.save_config(config_file)
        else:
            test_data = data.get_test_data()
            ncorrect = ai.evaluate(test_data)
            print(f"Correctly predicted test data: {100 * ncorrect / len(test_data)}%")

    if args.summarize_config:
        if config is None:
            config = cfg.from_file(config_file)
            print(config)



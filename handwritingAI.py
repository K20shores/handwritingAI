import numpy as np
import argparse

from mnist.ai import NeuralHandwritingNet
from mnist.utils import config as cfg
from mnist.utils.image_viewer import *
from mnist.data import get_test_data, get_training_data

def parse_args():
    parser = argparse.ArgumentParser(description='Configure the handwriting AI')
    parser.add_argument('-E', '--Epochs', type=int, default=30, dest='epochs',
            help='Number of epochs')
    parser.add_argument('-m', '--minibatchsize', type=int, default=10,
            help='Size of the minibatches')
    parser.add_argument('-e', '--eta', type=int, default=3,
            help='Eta, the learning rate')
    parser.add_argument('--hidden_layers', metavar='N', type=int, nargs='+', default=[100],
            help='A list of integers representing the number of nodes in each hidden layers.')
    parser.add_argument('-c', '--config', type=str, default='configs/config.conf',
            help='The config file to use to initialize the AI or save the result of a training run')
    parser.add_argument('-t', '--train', action='store_true',
            help='Use this to elect to train the AI. This will overwrite the config file if it exists.')
    parser.add_argument('-p', '--predict', action='store_true',
            help='Use this to elect to use the AI to predict values')
    parser.add_argument('-s', '--sumarize-config', action='store_true', dest='summarize_config',
            help='Use this to print out stats abot the config')
    parser.add_argument('-d', '--display', type=int, dest='display', default=None,
            help='Display the first N images in a grid. If both train and predit are specified, the first N images will be displayed.')
    parser.set_defaults(train=None)
    return parser.parse_args()

def display_data(data, n):
    images = []
    for i in range(n):
        images.append(np.reshape(data[i][0], (28, 28)))
    show_image_grid(images)

if __name__ == '__main__':
    args = parse_args()

    config_file = args.config

    config = None
    if args.train:
        layers = args.hidden_layers
        # input layers
        layers.insert(0, 28 * 28)
        # output layers
        layers.append(10)
        config = cfg.make_config(layers = layers, 
                eta = args.eta, 
                mini_batch_size = args.minibatchsize, 
                epochs = args.epochs)
    else:
        config = cfg.from_file(config_file)

    if args.summarize_config:
        if config is None:
            config = cfg.from_file(config_file)
        print(config)

    ai = NeuralHandwritingNet(config)

    data = None
    if args.train:
        data, test = get_training_data()
        if args.display is not None:
            display_data(data, args.display)

        ai.SGD(data, test_data=test)
        config.weights = ai.weights
        config.biases = ai.biases
        config.save_config(config_file)


    if args.predict:
        data = get_test_data()
        if args.display is not None:
            display_data(data, args.display)
        ncorrect = ai.evaluate(data)
        print(f"Correctly predicted test data: {100 * ncorrect / len(data)}%")



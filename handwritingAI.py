#!/usr/bin/env python
import numpy as np
import argparse
import os

from mnist.ai import NeuralHandwritingNet
from mnist.utils import config as cfg
from mnist.utils.image_viewer import show_image_grid
from mnist.data import get_test_data, get_training_data

def parse_args():
    parser = argparse.ArgumentParser(description='Configure the handwriting AI')
    parser.add_argument('-c', '--config', type=str, default='configs/config.conf',
            help='The config file to use to initialize the AI or save the result of a training run')
    parser.add_argument('-s', '--sumarize-config', action='store_true', dest='summarize_config',
            help='Use this to print out stats abot the config')

    training = parser.add_argument_group('training parameters')
    training.add_argument('-E', '--Epochs', type=int, default=30, dest='epochs',
            help='Number of epochs')
    training.add_argument('-m', '--minibatchsize', type=int, default=10,
            help='Size of the minibatches')
    training.add_argument('-e', '--eta', type=float, default=3,
            help='Eta, the learning rate')
    training.add_argument('-l', '--layers', metavar='N', type=int, nargs='+', default=[100],
            help='A list of integers representing the number of nodes in each hidden layers.')
    training.add_argument('-T', '--test-data', action='store_true', dest='test_data',
            help='Use this to elect to hold out 1/5th of the training data as test data to evaluate performance as the AI is trained')
    training.add_argument('-t', '--train', action='store_true',
            help='Use this to elect to train the AI. This will overwrite the config file if it exists.')

    predicting = parser.add_argument_group('predicting parameters')
    predicting.add_argument('-p', '--predict', action='store_true',
            help='Use this to elect to use the AI to predict values')

    image = parser.add_argument_group('image drawing parameters')
    image.add_argument('-n', type=int, dest='n_images', default=50, metavar='N',
            help='The number of images to pull from the dataset.')
    image.add_argument('-d', '--display', action='store_true', dest='display',
            help='If true, display the images in matplotlib.')
    image.add_argument('--image-display-offset', type=int, dest='image_display_offset', default=0,
            help='The number of images to skip when displaying images')
    image.add_argument('-C', '--color',  action='store_true',
            help='If displaying predicted images, color the images according to success or failure')
    image.add_argument('--save-image', type=str, dest='save_image', default=None,
            help='Will save the generated grid of images at the file location and name (type detected from extenstion). The file type must be valid for matplotlib. If an image is being generated for both training and predicting, -train and -predicted will be in the image path.')

    return parser.parse_args()

def display_data(data, color = False, results = None, save = False, file_path = None, show = False):
    data = [np.reshape(x[0], (28, 28)) for x in data]
    show_image_grid(data, color, results, save, file_path, show)

if __name__ == '__main__':
    args = parse_args()

    config_file = args.config

    config = None
    if args.train and not args.display:
        layers = args.layers
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

    if args.train:
        if args.display or args.save_image:
            # if we need to save the image and we are also going to save the predicted image,
            # add -train to the image path before the extension
            file_path = args.save_image
            save = False
            if args.save_image is not None:
                if args.predict:
                    name = os.path.basename(file_path)
                    file_parts = os.path.splitext(file_path)
                    file_path = f"{file_parts[0]}-train{file_parts[1]}"
                save = True
            data, test = get_training_data(args.n_images, args.image_display_offset)
            display_data(data, save=save, file_path=file_path, show=args.display)
        else:
            train, test = get_training_data(args.n_images, include_test_data=args.test_data)
            ai.SGD(train, test_data=test)
            config.weights = ai.weights
            config.biases = ai.biases
            config.save_config(config_file)

    if args.predict:
        if args.display or args.save_image:
            data = get_test_data(args.n_images, args.image_display_offset)
            ncorrect, results = ai.evaluate(data, progress_bar=False)
            # if we need to save the image and have saved the training image,
            # add -predicted to the image path before the extension
            file_path = args.save_image
            save = False
            if args.save_image is not None:
                if args.train:
                    name = os.path.basename(file_path)
                    file_parts = os.path.splitext(file_path)
                    file_path = f"{file_parts[0]}-predicted{file_parts[1]}"
                save = True
            display_data(data, args.color, results, save, file_path, show=args.display)
            print(f"Correctly predicted test data: {100 * ncorrect / len(data)}%")
        else:
            data = get_test_data()
            ncorrect, _ = ai.evaluate(data)
            print(f"Correctly predicted test data: {100 * ncorrect / len(data)}%")



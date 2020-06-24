import numpy as np
np.random.seed(0)
import json, codecs

def make_config(layers, eta, mini_batch_size, epochs):
    config = NeuralHandwritingNetConfig()
    config.layers = layers
    config.eta = eta
    config.mini_batch_size = mini_batch_size
    config.epochs = epochs
    config.biases = [np.random.randn(y, 1) for y in layers[1:]]
    config.weights = [np.random.randn(y, x)
                    for x, y in zip(layers[:-1], layers[1:])]
    return config

def from_file(file_name):
    config = NeuralHandwritingNetConfig()
    config.read_config(file_name)
    return config

class NeuralHandwritingNetConfig:
    def __init__(self):
        self.layers = None
        self.biases = None
        self.weights = None
        self.mini_batch_size = None
        self.epochs = None
        self.eta = None

    def __repr__(self):
        return f"Eta: {self.eta}, Epochs: {self.epochs}, Mini batch size: {self.mini_batch_size}, Layers: {self.layers}"

    def __str__(self):
        return f"Eta: {self.eta}, Epochs: {self.epochs}, Mini batch size: {self.mini_batch_size}, Layers: {self.layers}"

    def read_config(self, file_path):
        json_text = codecs.open(file_path, 'r', encoding='utf-8').read()
        json_data = json.loads(json_text)
        self.layers = json_data['layers']
        self.biases = [np.array(x) for x in json_data['biases']]
        self.weights = [np.array(x) for x in json_data['weights']]
        self.mini_batch_size = json_data['mini_batch_size']
        self.epochs = json_data['epochs']
        self.eta = json_data['eta']

    def save_config(self, file_path):
        layers = self.layers
        biases = [x.tolist() for x in self.biases]
        weights = [x.tolist() for x in self.weights]
        config = {
                'layers':layers,
                'biases': biases,
                'weights': weights,
                'mini_batch_size': self.mini_batch_size,
                'epochs': self.epochs,
                'eta': self.eta
            }
        json.dump(config,
                codecs.open(file_path, 'w', encoding='utf-8'),
                separators=(',', ':'), sort_keys=True, indent=4)


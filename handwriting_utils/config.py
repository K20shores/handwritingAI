import numpy as np
import json, codecs

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

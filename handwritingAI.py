import numpy as np
import struct
import math
from tqdm import tqdm

class LabelDataReader:
    def __init__(self, fileName):
        self.fileName = fileName

    def read(self):
        """ Read the data format. Returns a list of labels [0-9]

        Format is from MNIST http://yann.lecun.com/exdb/mnist/
        """

        labels = []

        with open(self.fileName, 'rb') as f:
            fileContents = f.read()

            offset = 0

            # grab the magic number to do a sanity check and the data length
            # see here for the format string, https://docs.python.org/3/library/struct.html#format-strings
            magic_number,data_length = struct.unpack(">ii", fileContents[offset:offset+8])
            offset += 8

            # this number is specified to be 2049 on the MNIST site above
            if magic_number != 2049:
                raise Exception("Invalid magic number. Make sure you are reading the number in big-endian format")

            print(f"Unpacking {data_length} label values")

            for offset in range(offset, len(fileContents)):
                label, = struct.unpack(">B", fileContents[offset:offset+1])
                labels.append(label)

            if (len(labels) != data_length):
                raise Exception(f"Failed to read the correct number of data values. Read {len(labels)}, expected {data_length}")
            print(f"Finished unpacking labels")

        return labels

class ImageDataReader:
    def __init__(self, fileName):
        self.fileName = fileName

    def read(self):
        """ Read the data format. Returns a list of np 2-d arrays

        Format is from MNIST http://yann.lecun.com/exdb/mnist/
        """

        images = []

        with open(self.fileName, 'rb') as f:
            fileContents = f.read()

            offset = 0

            # grab the magic number to do a sanity check, the number of images, rows, and columns
            # see here for the format string, https://docs.python.org/3/library/struct.html#format-strings
            magic_number,n_images, n_rows, n_columns = struct.unpack(">iiii", fileContents[offset:offset+16])
            offset += 16

            # this number is specified to be 2049 on the MNIST site above
            if magic_number != 2051:
                raise Exception("Invalid magic number. Make sure you are reading the number in big-endian format")

            print(f"Unpacking {n_images} images")
            print(f"Image size (row, column): ({n_rows}, {n_columns})")

            pixels = []
            for offset in range(offset, len(fileContents)):
                pixel, = struct.unpack(">B", fileContents[offset:offset+1])
                pixels.append(pixel)

            images = np.reshape(pixels, (n_images, n_columns * n_rows))


            if (len(images) != n_images):
                raise Exception(f"Failed to read the correct number of data values. Read {len(images)}, expected {n_images}")
            print(f"Finished unpacking images")

        return images

class NeuralHandwritingNet:
    def __init__(self):
        self.weights1 = np.random.rand(784,4)
        self.weights2 = np.random.rand(4,1)
        self.output = np.zeros(1)

    def train(self, x, y):
        self.input = x
        self.y = y

    def predict(self):
        pass

    def __backward_propogation(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * __sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * __sigmoid_derivative(self.output), self.weights2.T) * __sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def __feedforward(self):
        self.layer1 = __sigmoid(np.dot(self.input, self.weights1))
        self.output = __sigmoid(np.dot(self.layer1, self.weights2))

    #https://stackoverflow.com/a/29863846/5217293
    def __sigmoid(self, x):
        return math.exp(-np.logaddexp(0, -x))

    def __sigmoid_derivate(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

if __name__ == '__main__':
    ai = NeuralHandwritingNet()

    label_reader = LabelDataReader("training_data/train-labels-idx1-ubyte")
    labels = label_reader.read()

    image_reader = ImageDataReader("training_data/train-images-idx3-ubyte")
    images = image_reader.read()

    for (image, label) in tqdm(zip(images, labels), desc='Training'):
        ai.train(image, label)

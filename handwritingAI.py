import numpy as np
import struct
from tqdm import tqdm

class DataReader:
    def __init__(self, file_name, magic_number):
        self.file_name = file_name
        self.magic_number = magic_number
        with open(self.file_name, 'rb') as f:
            self.file_contents = f.read()
        self.offset = 0

    def read_header(self, fmt):
        sz = struct.calcsize(fmt)
        data = struct.unpack(fmt, self.file_contents[self.offset:self.offset+sz])
        self.offset += sz
        return data

    def check_magic_number(self, number):
        if self.magic_number != number:
            raise Exception("Invalid magic number. Make sure you are reading the number in big-endian format")

    def unpack_file_contents(self):
        return list(struct.iter_unpack(">B", self.file_contents[self.offset:]))

class LabelDataReader(DataReader):
    def __init__(self, file_name):
        super().__init__(file_name, 2049)

    def read(self):
        """ Read the data format. Returns a list of labels [0-9]

        Format is from MNIST http://yann.lecun.com/exdb/mnist/
        """

        magic_number,data_length = self.read_header(">ii")
        self.check_magic_number(magic_number)

        print(f"Unpacking {data_length} label values")
        labels = np.array(self.unpack_file_contents())
        print(f"Finished unpacking labels")

        if (len(labels) != data_length):
            raise Exception(f"Failed to read the correct number of data values. Read {len(labels)}, expected {data_length}")

        return labels

class ImageDataReader(DataReader):
    def __init__(self, file_name):
        super().__init__(file_name, 2051)

    def read(self):
        """ Read the data format. Returns a list of np 2-d arrays

        Format is from MNIST http://yann.lecun.com/exdb/mnist/
        """

        magic_number,n_images, n_rows, n_columns = self.read_header(">iiii")
        self.check_magic_number(magic_number)

        print(f"Unpacking {n_images} images")
        print(f"Image size (row, column): ({n_rows}, {n_columns})")
        images = np.reshape(self.unpack_file_contents(), (n_images, n_columns * n_rows))
        print(f"Finished unpacking images")

        if (len(images) != n_images):
            raise Exception(f"Failed to read the correct number of data values. Read {len(images)}, expected {n_images}")

        return images

class NeuralHandwritingNet:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.output = np.zeros(self.y.shape)

    def train(self):
        self.__feedforward()
        self.__backward_propogation()

    def predict(self, x):
        # TODO: is this correct? Probably not. Fix it.
        self.input = x
        self.__feedforward()
        return self.output

    def __feedforward(self):
        self.layer1 = self.__sigmoid(np.dot(self.input, self.weights1))
        self.output = self.__sigmoid(np.dot(self.layer1, self.weights2))

    def __backward_propogation(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        sse_derivative = 2*(self.y - self.output) 
        loss_derivative = self.__sigmoid_derivative(self.output)

        derivative_product = sse_derivative * loss_derivative

        d_weights2 = np.dot(self.layer1.T, derivative_product)

        layer2_gradient = np.dot(derivative_product, self.weights2.T) * self.__sigmoid_derivative(self.layer1)

        d_weights1 = np.dot(self.input.T,  layer2_gradient)

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    #https://stackoverflow.com/a/29863846/5217293
    def __sigmoid(self, x):
        return np.exp(-np.logaddexp(0, -x))

    def __sigmoid_derivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

if __name__ == '__main__':
    label_reader = LabelDataReader("training_data/train-labels-idx1-ubyte")
    labels = label_reader.read()[:1000]

    image_reader = ImageDataReader("training_data/train-images-idx3-ubyte")
    images = image_reader.read()[:1000]

    ai = NeuralHandwritingNet(images, labels)

    for i in tqdm(range(1500), desc='Training'):
        ai.train()

    print(ai.output)


import numpy as np
import struct
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    def read_all_contents(self):
        return list(struct.iter_unpack(">B", self.file_contents[self.offset:]))

    def read_n_bytes(self, n):
        data = struct.unpack(">" + "B" * n, self.file_contents[self.offset:self.offset+n])
        self.offset += n
        return data

class LabelDataReader(DataReader):
    def __init__(self, file_name):
        super().__init__(file_name, 2049)
        self.image_size = None

    def read(self, n = None):
        """ Read the data format. Returns a list of labels [0-9]

        Format is from MNIST http://yann.lecun.com/exdb/mnist/
        """

        magic_number,data_length = self.read_header(">ii")
        self.check_magic_number(magic_number)

        if n is not None:
            data_length = n

        print(f"Unpacking {data_length} label values")
        if n is None:
            labels = np.array([[x] for x in self.read_all_contents()])
        else:
            labels = np.array([[x] for x in self.read_n_bytes(n)])
        print(f"Finished unpacking labels")

        if (len(labels) != data_length):
            raise Exception(f"Failed to read the correct number of data values. Read {len(labels)}, expected {data_length}")

        return labels
        #return np.array([self.make_label_data(x) for x in labels])

    def make_label_data(self, x):
        data = ([0] * 10)
        data[x] = 1
        return data

class ImageDataReader(DataReader):
    def __init__(self, file_name):
        super().__init__(file_name, 2051)

    def read(self, n = None):
        """ Read the data format. Returns a list of numpy 1-d arrays

        Format is from MNIST http://yann.lecun.com/exdb/mnist/
        Each array contains all of the pixels for one image
        """

        magic_number,n_images, n_rows, n_columns = self.read_header(">iiii")
        self.image_size = (n_rows, n_columns)
        self.check_magic_number(magic_number)

        if n is not None:
            n_images = n

        print(f"Unpacking {n_images} images")
        print(f"Image size (row, column): ({self.image_size})")
        if n is None:
            images = np.reshape(self.read_all_contents(), (n_images, n_columns * n_rows)) / 255
        else:
            images = np.reshape(self.read_n_bytes(n * n_columns * n_rows), (n_images, n_columns * n_rows)) / 255
        print(f"Finished unpacking images")

        if (len(images) != n_images):
            raise Exception(f"Failed to read the correct number of data values. Read {len(images)}, expected {n_images}")

        return images

class NeuralHandwritingNet:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.weights1 = np.random.rand(self.input.shape[1],16)
        self.weights2 = np.random.rand(16,10)
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

    def __sigmoid(self, x):
        #could also use https://stackoverflow.com/a/29863846/5217293
        return 1 / (1 + np.exp(-x))

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

if __name__ == '__main__':
    n = int(1e4)
    label_reader = LabelDataReader("training_data/train-labels-idx1-ubyte")
    labels = label_reader.read(n)

    image_reader = ImageDataReader("training_data/train-images-idx3-ubyte")
    images = image_reader.read(n)
    image_size = image_reader.image_size

    #show_image(np.reshape(images[0], image_reader.image_size))
    #show_image_grid(np.reshape(images, (n, image_size[0], image_size[1])))

    ai = NeuralHandwritingNet(images, labels)

    for i in tqdm(range(int(1e3)), desc='Training'):
        ai.train()


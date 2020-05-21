import numpy as np
import struct

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

            # grab the magic number to do a sanity check (unpack returns a tuple, in this case it has one value. Hence, the comma
            # see here for the format string, https://docs.python.org/3/library/struct.html#format-strings
            magic_number,data_length = struct.unpack(">ii", fileContents[offset:offset+8])
            offset += 8

            # this number is specified to be 2049 on the MNIST site above
            if magic_number != 2049:
                raise Exception("Invalid magic number. Make sure you are reading the number in big-endian format")

            print(f"Unpacking {data_length} label values")

            for i in range(offset, len(fileContents)-3):
                label, = struct.unpack(">i", fileContents[i:i+4])
                labels.append(label)

            if (len(labels) != data_length):
                raise Exception(f"Failed to read the correct number of data values. Read {len(labels)}, expected {data_length}")

        return labels

class ImageDataReader:
    def __init__(self, fileName):
        self.fileName = fileName

    def read(self):
        pass

class NeuralHandwritingNet:
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def __backward_propogation(self):
        pass

    def __feedforward(self):
        pass

if __name__ == '__main__':
    ai = NeuralHandwritingNet()
    reader = LabelDataReader("training_data/train-labels-idx1-ubyte")
    print(reader.read())

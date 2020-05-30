import struct
import numpy as np
import gzip

class DataReader:
    def __init__(self, file_name, magic_number):
        self.file_name = file_name
        self.magic_number = magic_number
        with gzip.open(self.file_name, 'rb') as f:
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
            labels = np.array(self.read_all_contents())
        else:
            labels = np.array(self.read_n_bytes(n))
        print(f"Finished unpacking labels")

        if (len(labels) != data_length):
            raise Exception(f"Failed to read the correct number of data values. Read {len(labels)}, expected {data_length}")

        #return labels
        return np.array([self.make_label_data(x) for x in labels])

    def make_label_data(self, x):
        data = np.zeros((10, 1))
        data[x] = 1.0
        return data

class ImageDataReader(DataReader):
    def __init__(self, file_name):
        super().__init__(file_name, 2051)
        self.image_size = None

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
        shape = (n_images, n_columns * n_rows, 1)
        if n is None:
            images = np.reshape(self.read_all_contents(), shape) / 255
        else:
            images = np.reshape(self.read_n_bytes(n * n_columns * n_rows), shape) / 255
        print(f"Finished unpacking images")

        if (len(images) != n_images):
            raise Exception(f"Failed to read the correct number of data values. Read {len(images)}, expected {n_images}")

        return images


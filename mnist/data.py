from .utils.data_reader import LabelDataReader
from .utils.data_reader import ImageDataReader

def get_training_data(n = None, offset = 0, include_test_data = False):
    """Return a tuple of training and test data.

    If n is None, all data is read.
    Both training and test data is a tuple of image and label pairs.
    n is the number of images you want to read
    if test_data is True, then 1/5th of the number of training data will 
    be placed into test_data, otherwise None will be returned fro test_data
    """
    if n is not None:
        n = int(n)
    label_reader = LabelDataReader("mnist/training_data/train-labels-idx1-ubyte.gz")
    labels = label_reader.read(n, offset)

    image_reader = ImageDataReader("mnist/training_data/train-images-idx3-ubyte.gz")
    images = image_reader.read(n, offset)
    image_size = image_reader.image_size

    test_data = None
    if include_test_data:
        test_data_length = len(labels) // 5
        test_data_labels = labels[-test_data_length:]
        labels = labels[:-test_data_length]
        test_data_images = images[-test_data_length:]
        images = images[:-test_data_length:]
        test_data = [x for x in zip(test_data_images, test_data_labels)]

    training_data = [x for x in zip(images, labels)]

    return training_data, test_data

def get_test_data(n = None, offset = 0):
    """Return a tuple of image and label pairs

    If n is None, all data is read.
    """
    if n is not None:
        n = int(n)
    label_reader = LabelDataReader("mnist/test_data/t10k-labels-idx1-ubyte.gz")
    labels = label_reader.read(n, offset)

    image_reader = ImageDataReader("mnist/test_data/t10k-images-idx3-ubyte.gz")
    images = image_reader.read(n, offset)

    return [x for x in zip(images, labels)]

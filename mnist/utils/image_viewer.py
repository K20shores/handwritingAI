import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def show_image(data):
    """ Given a 2d array of pixel data, plot an mnist image

    The array is assumed to be an array of numbers with values 
    between 0 and 1. The image is plotted in greyscale.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def grey_to_rgba(data, color, successful = None, alpha = .3):
    # the website that provided the data says that
    # 0 corresponds to background (white) and 
    # 1 correspond to foreground (black) 
    # rgb data is the reverse of that, fix it here
    data = 1-data
    image = np.ones((28, 28, 4), dtype=np.float)
    image[:, :, 0] = data
    image[:, :, 1] = data
    image[:, :, 2] = data
    if color:
        mask = image[:, :, 0] == 1
        if successful is not None:
            if successful:
                image[:, :, 0][mask] = 0
                image[:, :, 1][mask] = 1
                image[:, :, 2][mask] = 0
            else:
                image[:, :, 0][mask] = 1
                image[:, :, 1][mask] = 0
                image[:, :, 2][mask] = 0
            image[:, :, 3][mask] = alpha
    return image

def show_image_grid(data, color = False, results = None, save = False, file_path = 'grid.svg', show = False):
    """ Given an 3d array of pixel data, plot an grid of mnist images

    The data is assumed to be a 3d array where the first dimension
    corresponds to the 2d array of pixel data that is an mnist image.
    """
    n = len(data)
    n_rows = int(np.ceil(np.sqrt(n)))
    n_cols = n_rows
    ax = plt.subplot(111)
    x_offset = 5
    y_offset = 5
    width = 28 * n_rows + n_rows * x_offset
    height = 28 * n_cols + n_cols * y_offset
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])

    for row in tqdm(range(n_rows), desc='Rendering images'):
        for col in range(n_cols):
            idx = n_cols * row + col
            if idx < n:
                success = None
                if results is not None:
                    success = results[idx]
                rgba = grey_to_rgba(data[idx], color, success)
                for x in range(28):
                    for y in range(28):
                        px = row*28 + x_offset*(row+1) + y
                        py = col*28 + y_offset*(col+1) + 28-x
                        r = rgba[x, y, 0]
                        g = rgba[x, y, 1]
                        b = rgba[x, y, 2]
                        a = rgba[x, y, 3]
                        ax.scatter(px, py, color = (r, g, b, a))

    ax.set_xticks([])
    ax.set_yticks([])
    if save:
        plt.savefig(file_path)
    if show:
        plt.show()


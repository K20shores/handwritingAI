import matplotlib.pyplot as plt
import numpy as np

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

def show_image_grid(data, color = False, results = None, save = False, file_path = 'grid.svg', show = False):
    """ Given an 3d array of pixel data, plot an grid of mnist images

    The data is assumed to be a 3d array where the first dimension
    corresponds to the 2d array of pixel data that is an mnist image.
    """
    images = []
    alpha = .3
    # convert the images to rgba
    # if we are coloring, color correct images green and 
    # incorrect images red
    for idx, im in enumerate(data):
        im = 1-im
        image = np.ones((28, 28, 4), dtype=np.float)
        image[:, :, 0] = im
        image[:, :, 1] = im
        image[:, :, 2] = im
        if color:
            successful = results[idx]
            mask = image[:, :, 0] == 1
            if successful:
                image[:, :, 0][mask] = 0
                image[:, :, 1][mask] = 1
                image[:, :, 2][mask] = 0
            else:
                image[:, :, 0][mask] = 1
                image[:, :, 1][mask] = 0
                image[:, :, 2][mask] = 0
            image[:, :, 3][mask] = alpha
        images.append(image)
    n = len(images)
    n_rows = int(np.ceil(np.sqrt(n)))
    n_cols = n_rows
    fig, axs = plt.subplots(n_rows, n_cols)

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axs[row][col]
            idx = n_cols * row + col
            if (idx >= n):
                ax.set_visible(False)
            else:
                ax.imshow(images[idx])
                ax.set_xticks([])
                ax.set_yticks([])
    if save:
        plt.savefig(file_path)
    if show:
        plt.show()


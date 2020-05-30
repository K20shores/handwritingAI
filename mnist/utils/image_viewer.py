import matplotlib.pyplot as plt


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

def show_image_grid(data):
    """ Given an 3d array of pixel data, plot an grid of mnist images

    The data is assumed to be a 3d array where the first dimension
    corresponds to the 2d array of pixel data that is an mnist image.
    """
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


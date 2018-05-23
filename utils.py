import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    notcar_image = mpimg.imread(notcar_list[0])
    data_dict["image_shape"] = notcar_image.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = notcar_image.dtype
    # Return data_dict
    return data_dict


def visualize(img_list, img_labels, title, rows, cols=2, fig_size=(15, 15), show_ticks=True):
    cmap = None

    fig, axes = plt.subplots(rows, cols, figsize=fig_size)

    for i in range(0, rows):
        for j in range(0, cols):
            ax = axes[i, j] if rows > 1 else axes[j]
            img_name = img_labels[i][j]
            img = img_list[i][j]

            if not show_ticks:
                ax.axis("off")

            ax.imshow(img, cmap=cmap)
            ax.set_title(img_name)

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1)
    fig.tight_layout()
    plt.show()
    return
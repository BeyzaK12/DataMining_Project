import os

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import segmentation, color
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.future import graph
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


def load_paths(folder_path_):
    global image_paths

    folders = os.listdir(folder_path_)
    for folder in folders:

        path_ = "{}/{}".format(folder_path_, folder)
        files_ = os.listdir(path_)

        for file_ in files_:
            img_path = "{}/{}".format(path_, file_)
            image_paths.append(img_path)


def transform_image(img_path_, size_):
    img = io.imread(img_path_)

    # img = rgb2gray(img)
    img = resize(img, (size_, size_), anti_aliasing=False)

    return img


def plot_original_transformed(no_, size_):
    img_path = image_paths[no_]
    io.imshow(img_path)
    plt.show()

    transformed_img = transform_image(image_paths[no_], size_)
    io.imshow(transformed_img)
    plt.show()


# https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html#sphx-glr-auto-examples-transform-plot-ssim-py
def structural_similarity_index(no_):
    img_path = image_paths[no_]
    img = io.imread(img_path)
    img = rgb2gray(img)

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    noise[np.random.random(size=noise.shape) > 0.5] *= -1

    img_noise = img + noise
    img_const = img + abs(noise)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    mse_none = mean_squared_error(img, img)
    ssim_none = ssim(img, img, data_range=img.max() - img.min())

    mse_noise = mean_squared_error(img, img_noise)
    ssim_noise = ssim(img, img_noise,
                      data_range=img_noise.max() - img_noise.min())

    mse_const = mean_squared_error(img, img_const)
    ssim_const = ssim(img, img_const,
                      data_range=img_const.max() - img_const.min())

    label = 'MSE: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_xlabel(label.format(mse_none, ssim_none))
    ax[0].set_title('Original image')

    ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
    ax[1].set_title('Image with noise')

    ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[2].set_xlabel(label.format(mse_const, ssim_const))
    ax[2].set_title('Image plus constant')

    plt.tight_layout()
    plt.show()


# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html#sphx-glr-auto-examples-segmentation-plot-thresholding-py
def thresholding(no_):
    img_path = image_paths[no_]
    img = io.imread(img_path)
    img = rgb2gray(img)

    noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    noise[np.random.random(size=noise.shape) > 0.5] *= -1

    img = img + abs(noise)

    thresh = threshold_otsu(img)
    binary = img > thresh

    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(img.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')

    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')

    plt.show()



# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_morphsnakes.html#sphx-glr-auto-examples-segmentation-plot-morphsnakes-py
# too bad
def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def morphological_snakes(no_):
    img_path = image_paths[no_]
    image = io.imread(img_path)
    image = rgb2gray(image)

    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3,
                                 iter_callback=callback)

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[7], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 7")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 35")
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)

    fig.tight_layout()
    plt.show()


folder_path = "tr_signLanguage_dataset"
image_paths = []

load_paths(folder_path)


image_no = 0
plot_original_transformed(no_=image_no , size_=64)

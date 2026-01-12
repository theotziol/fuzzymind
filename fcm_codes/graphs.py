import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from datetime import datetime

# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# def learning_plot(history, title = 'Training', c1 = 'b', c2 = 'c', save = True, name = '', show = True):
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     plt.plot(loss, color = c1, label = 'train_loss')
#     plt.plot(val_loss, color = c2, label = 'val_loss')
#     plt.legend()
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title(title)
#     if save:
#         now = datetime.now()
#         timestamp = int(datetime.timestamp(now))
#         plt.savefig(f'{timestamp}_{name}_loss.png', dpi = 600)
#     if show:
#         plt.show()
#     else:plt.close()

# def prediction_plot(actual, predicted, xticks_locs, xticks_labels,ylabel, save = True, name = '', show = True):
#     plt.plot(actual, label = 'Actual')
#     plt.plot(predicted, label = 'Predicted')
#     plt.title('Testing')
#     plt.xticks(xticks_locs, xticks_labels, rotation = 15)
#     plt.legend()
#     plt.xlabel('Time')
#     plt.ylabel(ylabel)
#     if save:
#         now = datetime.now()
#         timestamp = int(datetime.timestamp(now))
#         plt.savefig(f'{timestamp}_{name}_testing.png', dpi = 600)
#     if show:
#         plt.show()
#     else:plt.close()


# def calculate_and_plot_stats_of_matrices(list_of_predicted_matrices, index, title = ''):
#     '''
#     This function compares the statistics in the weight values among different runs by plotting them
#     Args:
#         list_of_predicted_matrices: list, that contais the weight_matrix predictions with shape (batch, h,w,c)
#         index: int, the index of the prediction to compare among runs
#         title: A title for the supplots graph
#     '''
#     _, h, w, _ = list_of_predicted_matrices[0].shape
#     c = len(list_of_predicted_matrices)
#     mean_weight_matrix = np.zeros((h,w,c))
#     for i in range(10):
#         mean_weight_matrix[:,:,i] = list_of_predicted_matrices[i][index,:,:,0]
#     fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     im, cbar = heatmap(np.mean(mean_weight_matrix, axis = -1), df.columns, df.columns,ax = ax, cmap = 'coolwarm')
#     texts = annotate_heatmap(im, size = 8)
#     ax.set_title('Average')

#     im, cbar = heatmap(np.std(mean_weight_matrix, axis = -1), df.columns, df.columns,ax = ax2, cmap = 'coolwarm')
#     texts = annotate_heatmap(im, size = 8)
#     ax2.set_title('std')

#     plt.tight_layout()
#     plt.suptitle(title)
#     plt.savefig(f'weight_matrices_water_average_std_index{index}.png', dpi = 1200)
#     plt.show()


def calculate_and_plot_stats_of_matrices(array, columns, y_test, title="", figsize = (12,5), size = 8):
    """
    This function compares the statistics in the weight values among the test dataset
    Args:
        array: weight_matrix predictions with shape (batch, h,w,c)
        index: int, the index of the prediction to compare among runs
        title: A title for the supplots graph
    """

    predicted_matrices = array.copy()
    b, h, w, _ = predicted_matrices.shape
    predicted_matrices = np.round(predicted_matrices, 2)
    for i in range(predicted_matrices.shape[1]):
        predicted_matrices[:, i, i, :] = 0.0
    for i in range(y_test.shape[-1]):
        predicted_matrices[:, -(i + 1), :, :] = 0.0
    mean_weight_matrix = np.zeros((h, w, b))

    for i in range(b):
        mean_weight_matrix[:, :, i] = predicted_matrices[i, :, :, 0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize)

    im, cbar = heatmap(
        np.mean(mean_weight_matrix, axis=-1), columns, columns, ax=ax, cmap="coolwarm"
    )
    texts = annotate_heatmap(im, size=size)
    ax.set_title("Average matrix")

    im, cbar = heatmap(
        np.std(mean_weight_matrix, axis=-1), columns, columns, ax=ax2, cmap="coolwarm"
    )
    texts = annotate_heatmap(im, size=size)
    ax2.set_title("std matrix")

    plt.tight_layout()
    plt.suptitle(title)
    # plt.savefig(f'weight_matrices_water_average_std_index{index}.png', dpi = 1200)
    # plt.show()
    return fig

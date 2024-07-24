import random

import pandas as pd
import numpy as np

import matplotlib as mpl
import pylab as pl
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from skimage.draw import line_aa

def random_color(nb=100):
    """
    Generate random color map
    :param nb: int number of different color
    :return:
    """

    get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
    color_list = get_colors(nb)
    color_list[0] = "#000000"
    mycolormap = mpl.colors.ListedColormap(color_list)
    return mycolormap


def colored_cell(image, cell_df, column, normalize=True, normalize_max=None, border=True, **kwargs):
    img_color = np.zeros(image.shape)
    color = cell_df[column].to_numpy()
    if normalize:
        if normalize_max is None:
            color = color/np.max(color)
        else:
            color = color/normalize_max

    for f, c in zip(cell_df["id_im"], color):
        pos = np.where(image == f)
        img_color[pos] = c

    return img_color

from skimage.draw import line_aa
def aniso_cell(image, cell_df, factor=2):
    img_color = np.zeros(image.shape)

    max_aniso = cell_df['aniso'].max()
    cell_df['norm_aniso'] = (cell_df['aniso'] / max_aniso)

    startx = (cell_df['x_center'] - factor * cell_df['norm_aniso'] * cell_df['orientation_y'])
    starty = (cell_df['y_center'] - factor * cell_df['norm_aniso'] * cell_df['orientation_x'])
    endx = (cell_df['x_center'] + factor * cell_df['norm_aniso'] * cell_df['orientation_y'])
    endy = (cell_df['y_center'] + factor * cell_df['norm_aniso'] * cell_df['orientation_x'])

    for i in cell_df.index:
        rr, cc, val = line_aa(int(starty[i]), int(startx[i]), int(endy[i]), int(endx[i]))
        img_color[rr, cc] = cell_df.loc[i]['norm_aniso'] * 100

    return img_color
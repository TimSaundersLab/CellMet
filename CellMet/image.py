import sparse

import pandas as pd
import numpy as np

from scipy import ndimage as ndi


def get_label(image, label):
    """
    Extract a mask where the label is.
    Parameters
    ----------
    image (np.array): label image
    label (int): value of the label

    Returns
    -------
    mask (np.array): mask where the label is.
    """
    return image == label


def get_unique_id_in_image(image, background_value=0):
    """
    Find all unique id in image.
    Parameters
    ----------
    image (np.array): label image
    background_value (int): default is 0

    Returns
    -------
    id_cells (list): list of all cell id in the image
    """
    id_cells = pd.unique(image.flatten())
    id_cells.sort()
    id_cells = np.delete(id_cells, np.where(id_cells == background_value))
    return id_cells


def find_neighbours_cell_id(img_cell_dil, img_seg, background_value=0, by_plane=False):
    """
    Find cell neighbours id of img_cell inside img_seg.
    Parameters
    ----------
    img_cell_dil (np.array): binary image of one dilated cell
    img_seg (np.array): label image
    background_value (int): default is 0

    Returns
    -------
    id_unique (list): list of all cell id that are neighbors to img_cell_id
    """
    img_multi = np.multiply(img_seg, img_cell_dil)
    id_unique = pd.unique(img_multi.flatten())
    id_unique = np.delete(id_unique, np.where(id_unique == background_value))

    if by_plane:
        nb_neighbors_plane = []
        for z in range(img_multi.shape[0]):
            id_unique_plane = pd.unique(img_multi[z].flatten())
            id_unique_plane = np.delete(id_unique_plane, np.where(id_unique_plane == 0))
            if len(id_unique_plane) == 0:
                nb_neighbors_plane.append(np.nan)
            else:
                nb_neighbors_plane.append(len(id_unique_plane))

        # remove all np.nan
        nb_neighbors_plane = np.array(nb_neighbors_plane)[~np.isnan(nb_neighbors_plane)]

        return id_unique, nb_neighbors_plane
    else:
        return id_unique, None


def find_cell_axis_center(img_cell, pixel_size, resize_image=True):
    """
    Calculate distance map, and keep the maximum distance in each z plane in order to find the center of the cell.
    Warning : result is slightly different from the measure made by ImageJ even if its is the same function that is used.

    Parameters
    ----------
    img_cell (np.array): binary image of one cell
    pixel_size (dict): size of pixel
    resize_image (bool): default True, allow to reduce np.array() size around the cell to fasten the distance_transform_edt calculation.

    Returns
    -------
    result (pd.DataFrame): position of the maximum distance for each z plane
    """
    if resize_image:
        # zz, yy, xx = np.where(img_cell == 1)
        sparce_cell = sparse.COO.from_numpy(img_cell)
        zz, yy, xx = sparce_cell.coords
        xx_min = max(xx.min() - 1, 0)
        yy_min = max(yy.min() - 1, 0)
        zz_min = max(zz.min() - 1, 0)

        xx_max = min(xx.max() + 1, img_cell.shape[2])
        yy_max = min(yy.max() + 1, img_cell.shape[1])
        zz_max = min(zz.max() + 1, img_cell.shape[0])

        sub_img_cell = img_cell[zz_min:zz_max,
                       yy_min:yy_max,
                       xx_min:xx_max
                       ]
    else:
        sub_img_cell = img_cell
        xx_min, yy_min, zz_min = 0, 0, 0

    edts = ndi.distance_transform_edt(sub_img_cell,
                                      sampling=[pixel_size["z_size"],
                                                pixel_size["y_size"],
                                                pixel_size["x_size"]],
                                      )
    # Find center (=highest value) in each plane
    x_c, y_c, z_c = [], [], []
    for k in range(edts.shape[0]):
        if edts[k].max() != 0:
            y_, x_ = np.where(edts[k] == edts[k].max())
            x_c.append(x_.mean() + xx_min)
            y_c.append(y_.mean() + yy_min)
            z_c.append(k + zz_min)

    result = pd.DataFrame(data=[np.array(x_c),
                                np.array(y_c),
                                np.array(z_c),
                                np.array(x_c) * pixel_size["x_size"],
                                np.array(y_c) * pixel_size["y_size"],
                                np.array(z_c) * pixel_size["z_size"],
                                ]).T
    result.columns = ["x_center", "y_center", "z_center",
                      "x_center_um", "y_center_um", "z_center_um"]
    return result


def colored_image_cell(image, cell_df, column, normalize=True, normalize_max=None):
    if column not in cell_df.columns:
        print("This columns does not exist : " + column)
        return

    if normalize:
        if normalize_max is None:
            cell_df[column + '_norm'] = cell_df[column] / cell_df[column].max()
        else:
            cell_df[column + '_norm'] = cell_df[column] / normalize_max

    colored_image = np.zeros(image.shape)

    for c in cell_df.index:
        cell_position = np.where(image == int(cell_df.loc[c]["id_im"]))
        if normalize:
            colored_image[cell_position] = cell_df.loc[c, column + '_norm']
        else:
            colored_image[cell_position] = cell_df.loc[c, column]

    return colored_image




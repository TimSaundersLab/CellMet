import os
import sparse

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull

from .segmentation import Segmentation
from . import utils as csutils
from . import image as csimage


def cell_analysis(seg: Segmentation):
    """

    :param seg : Segmentation object
    :return:
    """

    # Open cell_df
    cell_df = pd.read_csv(os.path.join(seg.storage_path, "cell_df.csv"),
                          index_col="Unnamed: 0")
    cell_plane_df = pd.read_csv(os.path.join(seg.storage_path, "cell_plane_df.csv"),
                                index_col="Unnamed: 0")

    # Calculate volume
    volume = []
    for c_id in cell_df['id_im']:
        sparse_cell = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
        volume.append(len(sparse_cell.coords[0]) * seg.voxel_size)
    cell_df["volume"] = volume

    # Calculate cell lengths and curvature
    real_dist = []
    short_dist = []
    curv_ind = []
    for c_id in cell_df['id_im']:
        df_ = cell_plane_df[cell_plane_df['id_im'] == c_id].copy()
        df_['x_center'] *= seg.pixel_size["x_size"]
        df_['y_center'] *= seg.pixel_size["y_size"]
        df_['z_center'] *= seg.pixel_size["z_size"]
        rd, sd, ci = calculate_lengths_curvature(df_, columns=["x_center", "y_center", "z_center"])
        real_dist.append(rd)
        short_dist.append(sd)
        curv_ind.append(ci)
    cell_df["real_dist"] = real_dist
    cell_df["short_dist"] = short_dist
    cell_df["curv_ind"] = curv_ind

    # Measure cell plane metrics
    aniso = []
    orientation_x = []
    orientation_y = []
    major = []
    minor = []
    area = []
    perimeter = []
    for c_id in cell_df['id_im']:
        sparse_cell = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
        img_cell = sparse_cell.todense()
        img_cell = csimage.get_label(img_cell, 1).astype("uint8")

        (a, ox, oy, maj, mi, ar, per) = measure_cell_plane(img_cell, seg.pixel_size)
        aniso.extend(a)
        orientation_x.extend(ox)
        orientation_y.extend(oy)
        major.extend(maj)
        minor.extend(mi)
        area.extend(ar)
        perimeter.extend(per)
    cell_plane_df["aniso"] = aniso
    cell_plane_df["orientation_x"] = orientation_x
    cell_plane_df["orientation_y"] = orientation_y
    cell_plane_df["major"] = major
    cell_plane_df["minor"] = minor
    cell_plane_df["area"] = area
    cell_plane_df["perimeter"] = perimeter

    calculate_cell_orientation(cell_df, cell_plane_df)

    cell_df.to_csv(os.path.join(seg.storage_path, "cell_df.csv"))
    cell_plane_df.to_csv(os.path.join(seg.storage_path, "cell_plane_df.csv"))


def edge_analysis(seg: Segmentation):
    """

    :param seg : Segmentation object
    :return:
    """

    edge_df = pd.read_csv(os.path.join(seg.storage_path, "edge_df.csv"),
                          index_col="Unnamed: 0")
    edge_pixel_df = pd.read_csv(os.path.join(seg.storage_path, "edge_pixel_df.csv"),
                                index_col="Unnamed: 0")

    # Calculate edge lengths and curvature
    real_dist = []
    short_dist = []
    curv_ind = []
    for id_, [c1, c2, c3] in edge_df[['id_im_1', 'id_im_2', 'id_im_3']].iterrows():
        df_ = edge_pixel_df[(edge_pixel_df['id_im_1'] == c1) &
                            (edge_pixel_df['id_im_2'] == c2) &
                            (edge_pixel_df['id_im_3'] == c3)].copy()
        df_['x'] *= seg.pixel_size["x_size"]
        df_['y'] *= seg.pixel_size["y_size"]
        df_['z'] *= seg.pixel_size["z_size"]
        df_ = df_.groupby('z').mean()
        df_.reset_index(drop=False, inplace=True)
        rd, sd, ci = calculate_lengths_curvature(df_, columns=list("xyz"))
        real_dist.append(rd)
        short_dist.append(sd)
        curv_ind.append(ci)
    edge_df["real_dist"] = real_dist
    edge_df["short_dist"] = short_dist
    edge_df["curv_ind"] = curv_ind

    # Calculate edge rotation around cell center
    rotation = []
    for id_, [c1, c2, c3] in edge_df[['id_im_1', 'id_im_2', 'id_im_3']].iterrows():
        df_ = edge_pixel_df[(edge_pixel_df['id_im_1'] == c1) &
                            (edge_pixel_df['id_im_2'] == c2) &
                            (edge_pixel_df['id_im_3'] == c3)].copy()
        df_['x_cell'] *= seg.pixel_size["x_size"]
        df_['y_cell'] *= seg.pixel_size["y_size"]
        df_['z_cell'] *= seg.pixel_size["z_size"]
        df_.reset_index(drop=False, inplace=True)
        rotation.append(0)
        for i in range(1, len(df_)):
            angle_rot = csutils.get_angle(
                (df_["x_cell"][i], df_["y_cell"][i]),
                (0, 0),
                (df_["x_cell"][i - 1], df_["y_cell"][i - 1]))
            rotation.append(angle_rot)
    edge_pixel_df["rotation"] = rotation

    edge_df.to_csv(os.path.join(seg.storage_path, "edge_df.csv"))
    edge_pixel_df.to_csv(os.path.join(seg.storage_path, "edge_pixel_df.csv"))


def face_analysis(seg: Segmentation):
    """

    :param seg : Segmentation object
    :return:
    """
    face_edge_pixel_df = pd.read_csv(os.path.join(seg.storage_path, "face_edge_pixel_df.csv"),
                                     index_col="Unnamed: 0")

    face_edge_pixel_df['length_um'] = np.sqrt(
        (face_edge_pixel_df['x_e1_mean'] - face_edge_pixel_df['x_e2_mean']) ** 2.0 + (
                face_edge_pixel_df['y_e1_mean'] - face_edge_pixel_df['y_e2_mean']) ** 2.0) * \
                                      seg.pixel_size[
                                          'x_size']
    face_edge_pixel_df['angle'] = (np.arctan2((face_edge_pixel_df['y_e2_mean'] * seg.pixel_size['y_size'] -
                                               face_edge_pixel_df['y_mid'] * seg.pixel_size[
                                                   'y_size']).to_numpy(),
                                              (face_edge_pixel_df['x_e2_mean'] * seg.pixel_size['x_size'] -
                                               face_edge_pixel_df['x_mid'] * seg.pixel_size[
                                                   'x_size']).to_numpy()) * 180 / np.pi)

    face_edge_pixel_df.to_csv(os.path.join(seg.storage_path, "face_edge_pixel_df.csv"))


def calculate_cell_orientation(cell_df, cell_plane_df, degree_convert=True):
    start = []
    end = []
    for i, val in cell_df.iterrows():
        start.append(
            cell_plane_df[cell_plane_df["id_im"] == val["id_im"]][['x_center', 'y_center', 'z_center']].to_numpy()[0])
        end.append(
            cell_plane_df[cell_plane_df["id_im"] == val["id_im"]][['x_center', 'y_center', 'z_center']].to_numpy()[-1])

    cell_df[['x_start', 'y_start', 'z_start']] = start
    cell_df[['x_end', 'y_end', 'z_end']] = end

    convert = 1
    if degree_convert:
        convert = 180 / np.pi
    cell_df['orient_zy'] = np.arctan2((cell_df["z_end"] - cell_df['z_start']).to_numpy(),
                                      (cell_df["y_end"] - cell_df['y_start']).to_numpy()) * convert
    cell_df['orient_xy'] = np.arctan2((cell_df["x_end"] - cell_df['x_start']).to_numpy(),
                                      (cell_df["y_end"] - cell_df['y_start']).to_numpy()) * convert
    cell_df['orient_xz'] = np.arctan2((cell_df["x_end"] - cell_df['x_start']).to_numpy(),
                                      (cell_df["z_end"] - cell_df['z_start']).to_numpy()) * convert


def calculate_lengths_curvature(data, columns):
    """
    Calculate length of the cell (real and shortest),  and curvature.
    Real distance is the distance between each pixel (we supposed to have one pixel per z plane).
    Shortest distance is the distance between the first and last plane of the cell.
    Curvature is calculated as $1-(short_dist/real_dist)$ .

    Parameters
    ----------
    data (pd.DataFrame): dataframe that contains position of the cell center
    columns (list): name of columns that contains position

    Returns
    -------
    real_dist (float)
    short_dist (float)
    curv_ind (float)

    """
    real_dist = np.linalg.norm(data.diff()[columns].values, axis=1)[1:].sum()
    short_dist = np.linalg.norm((data.iloc[-1] - data.iloc[0])[columns].values, )
    if (real_dist == 0) or (short_dist == 0):
        curv_ind = np.nan
    else:
        curv_ind = 1 - (short_dist / real_dist)

    return real_dist, short_dist, curv_ind


def measure_cell_plane(img_cell, pixel_size):
    """

    Parameters
    ----------
    img_cell (np.array): binary image
    pixel_size (dict): size of pixel

    Returns
    -------
    aniso (float): anisotropy of each z plane $major/minor$
    orientation_x, orientation_y (float, float): euler angle of the major axis
    major (float): length of the major axis
    minor (float): length of the minor axis
    area (float): in µm²
    perimeter (perimeter): in µm

    """
    aniso = []
    orientation_x = []
    orientation_y = []
    major = []
    minor = []
    area = []
    perimeter = []
    cell_in_z_plan = np.where(img_cell.sum(axis=1).sum(axis=1) > 0)[0]

    for z in cell_in_z_plan:
        points = np.array(np.where(img_cell[z, :, :] > 0)).flatten().reshape(len(np.where(img_cell[z, :, :] > 0)[1]),
                                                                             2,
                                                                             order='F')
        hull = ConvexHull(points)

        # Measure cell anisotropy
        # need to center the face at 0, 0
        # otherwise the calculation is wrong
        pts = (points - points.mean(axis=0)) * pixel_size['x_size']
        u, s, vh = np.linalg.svd(pts)
        svd = np.concatenate((s, vh[0, :]))

        s.sort()
        s = s[::-1]

        orientation = svd[2:]
        aniso.append(s[0] / s[1])
        orientation_x.append(orientation[0])
        orientation_y.append(orientation[1])
        major.append(s[0])
        minor.append(s[1])
        perimeter.append(hull.area)
        area.append(hull.volume)

    return aniso, orientation_x, orientation_y, major, minor, area, perimeter

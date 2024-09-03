import os
import sparse
import joblib

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull
from skimage.transform import resize
from .segmentation import Segmentation
from . import utils as csutils
from . import image as csimage


def simplified_cell_analysis(seg):
    """
    Only measure the volume to filter the segmentation
    :param seg:
    :return:
    """
    cell_df = pd.DataFrame(columns=["volume"])
    cell_df["id_im"] = seg.unique_id_cells
    for c_id in seg.unique_id_cells:
        sparse_cell = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
        img_cell_dil = sparse_cell.todense()
        img_cell_dil[img_cell_dil == 2] = 1
        img_cell = csimage.get_label(sparse_cell.todense(), 1).astype("uint8")
        volume = (len(sparse_cell.coords[0]) * seg.voxel_size)
        cell_df.loc[cell_df[cell_df["id_im"] == c_id].index, "volume"] = volume
        cell_df.to_csv(os.path.join(seg.storage_path, "cell_df.csv"))

def cell_analysis(seg: Segmentation, parallelized=True, degree_convert=True):
    """
    Analyse cell shape that only require one cell.
    :param seg: Segmentation object
    :param parallelized: bool to parallelized analysis
    :param degree_convert: bool to convert angle measure in degree
    :return:
    """
    cell_columns = ["id_im",
                    "volume",
                    "area",
                    "real_dist",
                    "short_dist",
                    "curv_ind",
                    "orient_zy",
                    "orient_xy",
                    "orient_xz",
                    'x_start',
                    'y_start',
                    'z_start',
                    'x_end',
                    'y_end',
                    'z_end',
                    ]
    if os.path.exists(os.path.join(seg.storage_path, "cell_df.csv")):
        cell_df = pd.read_csv(os.path.join(seg.storage_path, "cell_df.csv"),
                              index_col="Unnamed: 0")
        for c in cell_columns:
            if c not in cell_df.columns:
                cell_df[c] = np.nan
    else:
        cell_df = pd.DataFrame(columns=cell_columns)
        cell_df["id_im"] = seg.unique_id_cells

    cell_plane_columns = ["id_im",
                          "x_center",
                          "y_center",
                          "z_center",
                          "x_center_um",
                          "y_center_um",
                          "z_center_um",
                          "aniso",
                          "orientation_x",
                          "orientation_y",
                          "major",
                          "minor",
                          "area",
                          "perimeter",
                          "nb_neighbor"
                          ]
    # if os.path.exists(os.path.join(seg.storage_path, "cell_plane_df.csv")):
    #     cell_plane_df = pd.read_csv(os.path.join(seg.storage_path, "cell_plane_df.csv"),
    #                                 index_col="Unnamed: 0")
    #     for c in cell_plane_columns:
    #         if c not in cell_plane_df.columns:
    #             cell_plane_df[c] = np.nan
    # else:
    cell_plane_df = pd.DataFrame(columns=cell_plane_columns)


    if parallelized:
        delayed_call = [joblib.delayed(sc_analysis_parallel)(seg, int(c_id), degree_convert) for c_id in
                        seg.unique_id_cells]
        # res = joblib.Parallel(n_jobs=seg.nb_core)(delayed_call)
        with csutils.tqdm_joblib(desc="Cell analysis", total=len(seg.unique_id_cells)) as progress_bar:
            res = joblib.Parallel(n_jobs=seg.nb_core, prefer="threads")(delayed_call)

        for cell_out, cell_plane_out in res:
            for k in cell_out.keys():
                cell_df.loc[cell_df[cell_df["id_im"]==cell_out.get("id_im")].index[0], k] = cell_out.get(k)
            cell_plane_df = pd.concat([cell_plane_df, pd.DataFrame(cell_plane_out,  columns=cell_plane_columns)], ignore_index=True)
    else:
        for c_id in seg.unique_id_cells:
            res = sc_analysis_parallel(seg, int(c_id))
            for k in res[0].keys():
                cell_df.loc[cell_df[cell_df["id_im"] == res[0].get("id_im")].index[0], k] = res[0].get(k)
            cell_plane_df = pd.concat([cell_plane_df, pd.DataFrame(res[1], columns=cell_plane_columns)], ignore_index=True)

    # 1 for a sphere
    cell_df['sphericity'] = (np.pi ** (1 / 3) * (6 * cell_df["volume"]) ** (2 / 3)) / cell_df["area"]
    # Save dataframe
    cell_df.to_csv(os.path.join(seg.storage_path, "cell_df.csv"))
    cell_plane_df.to_csv(os.path.join(seg.storage_path, "cell_plane_df.csv"))


def sc_analysis_parallel(seg, c_id, degree_convert=True):
    # open image
    sparse_cell = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
    img_cell_dil = sparse_cell.todense()
    img_cell_dil[img_cell_dil == 2] = 1
    img_cell = csimage.get_label(sparse_cell.todense(), 1).astype("uint8")
    data_ = csimage.find_cell_axis_center(img_cell, seg.pixel_size, resize_image=True)
    # measure nb neighbours
    neighbours_id, nb_neighbors_plane = csimage.find_neighbours_cell_id(img_cell_dil, seg.label_image, by_plane=True)
    (a, ox, oy, maj, mi, ar, per) = measure_cell_plane(img_cell, seg.pixel_size)

    cell_plane_out = np.array([np.repeat(c_id, len(data_["x_center"].to_numpy())),
                               data_["x_center"].to_numpy(),
                               data_["y_center"].to_numpy(),
                               data_["z_center"].to_numpy(),
                               data_["x_center_um"].to_numpy(),
                               data_["y_center_um"].to_numpy(),
                               data_["z_center_um"].to_numpy(),
                               a, ox, oy, maj, mi, ar, per,
                               nb_neighbors_plane.T,
                               ]).T
    start = data_[["x_center", "y_center", "z_center"]].to_numpy()[0]
    end = data_[["x_center", "y_center", "z_center"]].to_numpy()[-1]
    convert = 1
    if degree_convert:
        convert = 180 / np.pi
    orient_zy = np.arctan2((end[2] - start[2]),
                           (end[1] - start[1])) * convert
    orient_xy = np.arctan2((end[0] - start[0]),
                           (end[1] - start[1])) * convert
    orient_xz = np.arctan2((end[0] - start[0]),
                           (end[2] - start[2])) * convert

    volume = (len(sparse_cell.coords[0]) * seg.voxel_size)
    img_resize = resize(sparse_cell.todense() == 2,
                        (int(sparse_cell.shape[0] * seg.pixel_size["z_size"] / seg.pixel_size["x_size"]),
                         sparse_cell.shape[1],
                         sparse_cell.shape[2]))
    area = np.count_nonzero(img_resize == 1) * seg.pixel_size["x_size"]

    rd, sd, ci = calculate_lengths_curvature(data_, columns=["x_center_um", "y_center_um", "z_center_um"])
    cell_out = {"id_im": int(c_id),
                "volume": volume,
                "area": area,
                "real_dist": rd,
                "short_dist": sd,
                "curv_ind": ci,
                "orient_zy": orient_zy,
                "orient_xy": orient_xy,
                "orient_xz": orient_xz,
                'x_start': start[0],
                'y_start': start[1],
                'z_start': start[2],
                'x_end': end[0],
                'y_end': end[1],
                'z_end': end[2],
                }

    return cell_out, cell_plane_out


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


    face_df = pd.read_csv(os.path.join(seg.storage_path, "face_df.csv"),
                          index_col="Unnamed: 0")
    face_pixel_df = pd.read_csv(os.path.join(seg.storage_path, "face_pixel_df.csv"),
                                     index_col="Unnamed: 0")
    # area calculation
    df = face_edge_pixel_df.groupby(["id_im_1", "id_im_2", "edge_1", "edge_2"])["length_um"].sum() * seg.pixel_size["z_size"]
    df.index = df.index.set_names(["id_im_1", "id_im_2", "edge_1", "edge_2"])
    df = df.reset_index()
    face_df["area"] = np.nan
    for i, val in df.iterrows():
        f_id = face_df[((face_df["id_im_1"] == val["id_im_1"]) & (face_df["id_im_2"] == val["id_im_2"])) |
                       (face_df["id_im_1"] == val["id_im_2"]) & (face_df["id_im_2"] == val["id_im_1"])].index
        face_df.loc[f_id, 'area'] = val["length_um"]

    # perimeter calculation
    df = 2 * (face_edge_pixel_df.groupby(["id_im_1", "id_im_2", "edge_1", "edge_2"])["length_um"].mean() +
              face_edge_pixel_df.groupby(["id_im_1", "id_im_2", "edge_1", "edge_2"])["length_um"].count()*seg.pixel_size["z_size"])
    df.index = df.index.set_names(["id_im_1", "id_im_2", "edge_1", "edge_2"])
    df = df.reset_index()
    face_df["perimeter"] = np.nan
    for i, val in df.iterrows():
        f_id = face_df[((face_df["id_im_1"] == val["id_im_1"]) & (face_df["id_im_2"] == val["id_im_2"])) |
                       (face_df["id_im_1"] == val["id_im_2"]) & (face_df["id_im_2"] == val["id_im_1"])].index
        face_df.loc[f_id, 'perimeter'] = val["length_um"]

    face_df.to_csv(os.path.join(seg.storage_path, "face_df.csv"))



# def calculate_cell_orientation(cell_df, cell_plane_df, degree_convert=True):
#     start = []
#     end = []
#     for i, val in cell_df.iterrows():
#         start.append(
#             cell_plane_df[cell_plane_df["id_im"] == val["id_im"]][['x_center', 'y_center', 'z_center']].to_numpy()[0])
#         end.append(
#             cell_plane_df[cell_plane_df["id_im"] == val["id_im"]][['x_center', 'y_center', 'z_center']].to_numpy()[-1])
#
#     cell_df[['x_start', 'y_start', 'z_start']] = start
#     cell_df[['x_end', 'y_end', 'z_end']] = end
#
#     convert = 1
#     if degree_convert:
#         convert = 180 / np.pi
#     cell_df['orient_zy'] = np.arctan2((cell_df["z_end"] - cell_df['z_start']).to_numpy(),
#                                       (cell_df["y_end"] - cell_df['y_start']).to_numpy()) * convert
#     cell_df['orient_xy'] = np.arctan2((cell_df["x_end"] - cell_df['x_start']).to_numpy(),
#                                       (cell_df["y_end"] - cell_df['y_start']).to_numpy()) * convert
#     cell_df['orient_xz'] = np.arctan2((cell_df["x_end"] - cell_df['x_start']).to_numpy(),
#                                       (cell_df["z_end"] - cell_df['z_start']).to_numpy()) * convert


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
        if len(points)>10:
            try:
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
            except:
                aniso.append(np.nan)
                orientation_x.append(np.nan)
                orientation_y.append(np.nan)
                major.append(np.nan)
                minor.append(np.nan)
                perimeter.append(np.nan)
                area.append(np.nan)

        else:
            aniso.append(np.nan)
            orientation_x.append(np.nan)
            orientation_y.append(np.nan)
            major.append(np.nan)
            minor.append(np.nan)
            perimeter.append(np.nan)
            area.append(np.nan)


    return aniso, orientation_x, orientation_y, major, minor, area, perimeter

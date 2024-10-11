import os
import sparse
import joblib
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import networkx as nx

from scipy import ndimage as ndi

from . import utils as csutils
from . import image as csimage
from . import io as csio
from .segmentation import Segmentation, edge_detection, find_all_neighbours

class Segmentation2D(Segmentation):
    def __init__(self, image=None, pixel_size=None, path=None, nb_core=None):
        """Segmentation class constructor
        """

        if image is not None:
            self.label_image = image
            self.unique_id_cells = csimage.get_unique_id_in_image(image)
        else:
            self.label_image = np.empty(0)
            self.unique_id_cells = np.empty(0)

        if pixel_size is not None:
            self.pixel_size = pixel_size
        else:
            self.pixel_size = dict(x_size=1, y_size=1)
        self.voxel_size = np.prod(list(self.pixel_size.values()))
        self.struct_dil = csutils.generate_struct_dil(dim=2)

        if path is not None:
            self.storage_path = path
        else:
            self.storage_path = ""

        if nb_core is None:
            self.nb_core = os.cpu_count() - 2
        else:
            self.nb_core = nb_core

    def cell_segmentation(self):
        """
        Analyse cell parameter such as volume, length, number of neighbor.
        Calculate also plane by plane area, orientation perimeter...
        Parameters
        ----------

        Returns
        -------
        cell_df (pd.DataFrame): result for each cells
        cell_plane_df (pd.DataFrame): result for each plane of each cell
        """

        cell_columns = ["id_im",
                        "x_center",
                        "y_center",
                        "nb_neighbor",
                        ]
        cell_df = pd.DataFrame(columns=cell_columns)

        # for c_id in self.unique_id_cells:
        for i in tqdm(range(len(self.unique_id_cells)), desc="Cell", leave=True):
            c_id = self.unique_id_cells[i]
            c_id = int(c_id)
            # open image
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell_dil = sp_mat.todense()
            img_cell = csimage.get_label(img_cell_dil, 1).astype("uint8")
            img_cell_dil[img_cell_dil == 2] = 1


            # measure nb neighbours
            neighbours_id, nb_neighbors_plane = csimage.find_neighbours_cell_id(img_cell_dil, self.label_image, by_plane=True)

            # Get center of the cell
            sparce_cell = sparse.COO.from_numpy(img_cell)
            y, x = sparce_cell.coords.mean(axis=1)

            # Populate cell dataframe
            cell_df.loc[len(cell_df)] = {"id_im": int(c_id),
                                         "x_center": x,
                                         "y_center": y,
                                         "nb_neighbor": len(neighbours_id) - 1,
                                         }
            cell_df.to_csv(os.path.join(self.storage_path, "cell_df.csv"))


    def vert_segmentation(self):
        """

        :return:
        """

        # Open cell dataframe
        cell_df = pd.read_csv(os.path.join(self.storage_path, "cell_df.csv"))

        vert_columns = ["id_im_1",
                        "id_im_2",
                        "id_im_3",
                        "x", "y"
                        ]
        vert_df = pd.DataFrame(columns=vert_columns)

        # for c_id in self.unique_id_cells:
        for i in tqdm(range(len(self.unique_id_cells)), desc="Cell", leave=True):
            c_id = self.unique_id_cells[i]
            # step 1
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell1_dil = sp_mat.todense()

            img_cell1_dil[img_cell1_dil == 2] = 1

            neighbours_id, _ = csimage.find_neighbours_cell_id(img_cell1_dil, self.label_image, by_plane=False, background_value=-1)

            cell_combi = csutils.make_all_list_combination(np.delete(neighbours_id, np.where(c_id == neighbours_id)),
                                                           2)

            delayed_call = [
                joblib.delayed(vert_detection_2d)(self, cell_df, c_id, img_cell1_dil,
                                               cb, cc)
                for cb, cc in cell_combi]
            with csutils.tqdm_joblib(desc="Vert segmentation", total=len(cell_combi)) as progress_bar:
                res = joblib.Parallel(n_jobs=self.nb_core, prefer="threads")(delayed_call)

            for e_dict in res:
                vert_df.loc[len(vert_df)] = e_dict

            vert_df.to_csv(os.path.join(self.storage_path, "vert_df.csv"))
        vert_df.dropna(inplace=True)
        vert_df[["id_im_1", "id_im_2", "id_im_3"]] = np.sort(vert_df[["id_im_1", "id_im_2", "id_im_3"]])
        vert_df.drop_duplicates(["id_im_1", "id_im_2", "id_im_3"], inplace=True)
        vert_df.reset_index(inplace=True, drop=True)
        vert_df.to_csv(os.path.join(self.storage_path, "vert_df.csv"))


    def edge_segmentation(self):
        """

        :return:
        """
        vert_df = pd.read_csv(os.path.join(self.storage_path, "vert_df.csv"))

        edge_columns = ["id_im_1", "id_im_2",
                        "vert_1", "vert_2",
                        "x", "y",
                        ]
        edge_pixel_df = pd.DataFrame(columns=edge_columns)
        edge_df = pd.DataFrame(columns=["id_im_1", "id_im_2",
                                        "vert_1", "vert_2",
                                        ], )


        for i in tqdm(range(len(self.unique_id_cells)), desc="Cell", leave=True):
            c_id = self.unique_id_cells[i]
            # open file
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell_dil = sp_mat.todense()
            img_cell_dil[img_cell_dil == 2] = 1

            neighbours_id, _ = csimage.find_neighbours_cell_id(img_cell_dil, self.label_image, by_plane=False,
                                                               background_value=-1)
            neighbours_id = np.delete(neighbours_id, np.where(neighbours_id == c_id))
            # sub_edges = vert_df[vert_df['id_im_1'] == c_id]

            # ordered_neighbours, opp_cell = find_all_neighbours(sub_edges)

            if len(neighbours_id) != 0:
                delayed_call = [
                    joblib.delayed(edge_detection_2d)(self.storage_path, c_id, c_op_index, "",
                                                      img_cell_dil,
                                                   "", edge_df.columns, edge_pixel_df.columns,
                                                   edge_pixel_df, vert_df, self
                                                   )
                    for c_op_index in neighbours_id]
                # res = joblib.Parallel(n_jobs=self.nb_core)(delayed_call)
                with csutils.tqdm_joblib(desc="Face segmentation", total=len(neighbours_id)) as progress_bar:
                    res = joblib.Parallel(n_jobs=self.nb_core, prefer="threads")(delayed_call)

                res = [(f_pixel, f_dict) for f_pixel, f_dict in res if f_pixel is not None]
                for f_pixel, f_dict in res:
                    edge_pixel_df = pd.concat([df for df in [edge_pixel_df, f_pixel] if not df.empty],
                                              ignore_index=True)
                    edge_df.loc[len(edge_df)] = f_dict

#             edge_df.drop_duplicates(["id_im_1", "id_im_2", "vert_1", "vert_2"], inplace=True)
            edge_df.to_csv(os.path.join(self.storage_path, "edge_df.csv"))
        edge_pixel_df.to_csv(os.path.join(self.storage_path, "edge_pixel_df.csv"))



def vert_detection_2d(seg, cell_df, c_id, img_cell1_dil, cb, cc ):
    sp_mat = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(int(cb)) + ".npz"))
    img_cell_b_dil = sp_mat.todense()
    img_cell_b_dil[img_cell_b_dil == 2] = 1

    sp_mat = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(int(cc)) + ".npz"))
    img_cell_c_dil = sp_mat.todense()
    img_cell_c_dil[img_cell_c_dil == 2] = 1

    img_vert = np.multiply(np.multiply(img_cell1_dil, img_cell_b_dil), img_cell_c_dil)

    if len(pd.unique(img_vert.flatten())) < 2:
        return np.repeat(None,5)
    # orient cell counterclockwise
    # need for lateral face analyses
    a_ = csutils.get_angle(
        (cell_df[cell_df['id_im'] == cb]['x_center'].to_numpy()[0] * seg.pixel_size['x_size'],
         cell_df[cell_df['id_im'] == cb]['y_center'].to_numpy()[0] * seg.pixel_size['y_size']),
        (
            cell_df[cell_df['id_im'] == c_id]['x_center'].to_numpy()[0] * seg.pixel_size[
                'x_size'],
            cell_df[cell_df['id_im'] == c_id]['y_center'].to_numpy()[0] * seg.pixel_size[
                'y_size']),
        (cell_df[cell_df['id_im'] == cc]['x_center'].to_numpy()[0] * seg.pixel_size['x_size'],
         cell_df[cell_df['id_im'] == cc]['y_center'].to_numpy()[0] * seg.pixel_size['y_size']))
    if a_ > 0:
        cc, cb = cb, cc

    x0, y0 = np.mean(np.where(img_vert>0), axis=1)

    e_dict = {"id_im_1": int(c_id),
              "id_im_2": int(cb),
              "id_im_3": int(cc),
              "x": x0,
              "y": y0,
              }

    return e_dict


def edge_detection_2d(path, c_id, c_op_index, opp_cell, img_cell_dil, sub_edges, edge_columns, edge_pixel_columns,
                   edge_pixel_df, vert_df, seg):

    # c_op, a, b, c, d = opp_cell[c_op_index]
    c_op = c_op_index
    sp_mat = sparse.load_npz(os.path.join(path, "npz/" + str(int(c_op)) + ".npz"))
    img_cell_dil2 = sp_mat.todense()
    img_cell_dil2[img_cell_dil2 == 2] = 1
    img_edge = np.multiply(img_cell_dil, img_cell_dil2)

    sparce_edge = sparse.COO.from_numpy(img_edge)
    y0, x0 = sparce_edge.coords

    img_edge_dil = ndi.binary_dilation(img_edge, structure=seg.struct_dil)
    img_cell_n = np.multiply(img_edge_dil, seg.label_image)
    cell_id_n = pd.unique(img_cell_n.flatten())
    cell_id_n = np.delete(cell_id_n, np.where(cell_id_n == c_id))
    cell_id_n = np.delete(cell_id_n, np.where(cell_id_n == c_op))
    # print(cell_id_n)
    if len(cell_id_n)>2:
        print(c_id, c_op, cell_id_n)
        cpt=0
        for c in cell_id_n:
            c1, c2, c3 = np.sort([c_id, c_op, c])
            if len(vert_df[((vert_df["id_im_1"]==c1) & (vert_df["id_im_2"]==c2) & (vert_df["id_im_3"]==c3))])>0:
                if cpt==0:
                    a = c
                    cpt+=1
                elif cpt==1:
                    b=c
                    cpt+=1
                # else :
                #     c1, c2, c3 = np.sort([c_id, c_op, a])
                #     c4, c5, c6 = np.sort([c_id, c_op, c])
                #     c7, c8, c9 = np.sort([c_id, c_op, b])
                #     if not in_circle(vert_df[(vert_df["id_im_1"]==c1) & (vert_df["id_im_2"]==c2) & (vert_df["id_im_3"]==c3)]["x"].to_numpy()[0],
                #               vert_df[(vert_df["id_im_1"]==c1) & (vert_df["id_im_2"]==c2) & (vert_df["id_im_3"]==c3)]["y"].to_numpy()[0],
                #               0.5,
                #               vert_df[(vert_df["id_im_1"] == c4) & (vert_df["id_im_2"] == c5) & (vert_df["id_im_3"] == c6)]["x"].to_numpy()[0],
                #               vert_df[(vert_df["id_im_1"] == c4) & (vert_df["id_im_2"] == c5) & (vert_df["id_im_3"] == c6)]["y"].to_numpy()[0],
                #               ):
                #         a = c
                #
                #     elif not in_circle(vert_df[(vert_df["id_im_1"] == c7) & (vert_df["id_im_2"] == c8) & (vert_df["id_im_3"] == c9)]["x"].to_numpy()[0],
                #                  vert_df[(vert_df["id_im_1"] == c7) & (vert_df["id_im_2"] == c8) & (vert_df["id_im_3"] == c9)]["y"].to_numpy()[0],
                #                  0.5,
                #                  vert_df[(vert_df["id_im_1"] == c4) & (vert_df["id_im_2"] == c5) & (vert_df["id_im_3"] == c6)]["x"].to_numpy()[0],
                #                  vert_df[(vert_df["id_im_1"] == c4) & (vert_df["id_im_2"] == c5) & (vert_df["id_im_3"] == c6)]["y"].to_numpy()[0],
                #                  ):
                #         b = c
    elif len(cell_id_n) == 2:
        a = cell_id_n[0]
        b = cell_id_n[1]
    elif len(cell_id_n)==1:
        a = cell_id_n[0]
        b = None
    elif len(cell_id_n) == 0:
        a = None
        b = None

    e1 = pd.DataFrame.from_dict({
            'x': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                               (edge_pixel_df['id_im_2'] == c_op)]['x'].to_numpy(),
            'y': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                               (edge_pixel_df['id_im_2'] == c_op)]['y'].to_numpy(),
                              })

    f_pixel = pd.DataFrame(np.array([np.repeat(c_id, len(np.array(x0))),
                                     np.repeat(c_op, len(np.array(x0))),
                                     np.repeat(a, len(np.array(x0))),
                                     np.repeat(b, len(np.array(x0))),
                                     np.array(x0),
                                     np.array(y0),
                                     ]).T,
                           columns=edge_pixel_columns)

    if a is None:
        v1 = None
    else:
        try:
            c1, c2, c3 = np.sort([c_id, c_op, a])
            v1 = vert_df[((vert_df["id_im_1"]==c1) & (vert_df["id_im_2"]==c2) & (vert_df["id_im_3"]==c3))].index[0]
        except:
            v1 = None

    if b is None:
        v2 = None
    else:
        try:
            c1, c2, c3 = np.sort([c_id, c_op, b])
            v2 = vert_df[((vert_df["id_im_1"]==c1) & (vert_df["id_im_2"]==c2) & (vert_df["id_im_3"]==c3))].index[0]
        except:
            v2 = None

    f_dict = {"id_im_1": c_id,
              "id_im_2": c_op,
              "vert_1": v1,
              "vert_2": v2,
              }


    return f_pixel, f_dict


def cell_analysis_2d(seg: Segmentation, parallelized=True, degree_convert=True):
    """
    Analyse cell shape that only require one cell.
    :param seg: Segmentation object
    :param parallelized: bool to parallelized analysis
    :return:
    """
    cell_columns = ["id_im",
                    "aniso",
                    "orientation_x",
                    "orientation_y",
                    "major",
                    "minor",
                    "area",
                    "perimeter"
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

    if parallelized:
        delayed_call = [joblib.delayed(sc_analysis_parallel_2d)(seg, int(c_id), degree_convert) for c_id in
                        seg.unique_id_cells]
        # res = joblib.Parallel(n_jobs=seg.nb_core)(delayed_call)
        with csutils.tqdm_joblib(desc="Cell analysis", total=len(seg.unique_id_cells)) as progress_bar:
            res = joblib.Parallel(n_jobs=seg.nb_core, prefer="threads")(delayed_call)

        for cell_out in res:
            cell_df.loc[cell_df[cell_df["id_im"] == cell_out[0][0]].index[0], cell_columns] = cell_out[0]
    else:
        for c_id in seg.unique_id_cells:
            res = sc_analysis_parallel_2d(seg, int(c_id))
            cell_df.loc[cell_df[cell_df["id_im"] == res[0][0]].index[0]] = res

    # Save dataframe
    cell_df.to_csv(os.path.join(seg.storage_path, "cell_df.csv"))


def sc_analysis_parallel_2d(seg, c_id, degree_convert=True):
    # open image
    sparse_cell = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))

    img_cell = csimage.get_label(sparse_cell.todense(), 1).astype("uint8")

    (a, ox, oy, maj, mi, ar, per) = measure_cell_plane_2d(img_cell, seg.pixel_size)

    cell_plane_out = np.array([[int(c_id)],
                               a, ox, oy, maj, mi, ar, per,
                               ]).T

    return cell_plane_out


from scipy.spatial import ConvexHull


def measure_cell_plane_2d(img_cell, pixel_size):
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

    points = np.array(np.where(img_cell > 0)).flatten().reshape(len(np.where(img_cell > 0)[1]),
                                                                2,
                                                                order='F')

    if len(points) > 10:
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
            orientation_x.append(orientation[1])
            orientation_y.append(orientation[0])
            major.append(s[0])
            minor.append(s[1])
            perimeter.append(hull.area)
            area.append(len(points) * pixel_size['x_size'] * pixel_size['y_size'])
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


def in_circle(center_x, center_y, radius, x, y):
    """
    center_x , center_y are the coordinate of the circle center
    radius, is the radius of the circle
    x,y are the coordinate of the point you want to know if it is inside the circle or not
    """
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2
import os
import sparse

import numpy as np
import pandas as pd
import networkx as nx

from scipy import ndimage as ndi

from . import utils as csutils
from . import image as csimage


class Segmentation:

    def __init__(self):
        self.label_image = np.array()
        self.pixel_size = dict(x_size=1, y_size=1, z_size=1)
        self.voxel_size = np.prod(list(self.pixel_size.values()))
        self.struct_dil = None
        self.unique_id_cells = np.array()
        self.storage_path = ""

    def __init__(self, image, pixel_size, path):
        self.label_image = image
        self.pixel_size = pixel_size
        self.voxel_size = np.prod(list(self.pixel_size.values()))
        self.struct_dil = csutils.generate_struct_dil()
        self.unique_id_cells = csimage.get_unique_id_in_image(image)
        self.storage_path = path

    def perform_prerequisite(self, overright=False, save_mesh=False):
        """
        Perform prerequisite that is required for further analysis.
        This fonction :
            - find the center of each cell (in z(depth) axis)
            - save one sparse matrix per cell that contains its pixel and dilated
        Parameters
        ----------
        overright (bool): default False, allow to over right if directory "Center_fibre_coord" is not empty
        save_obj (bool): default False, choose to save mesh file

        Returns
        -------

        """
        if not os.path.exists(self.storage_path):
            os.mkdir(self.storage_path)

        directories = ["obj_mesh", "npz"]
        for d in directories:
            if not os.path.exists(self.storage_path + d):
                os.mkdir(self.storage_path + d, )

            list_dir = os.listdir(self.storage_path + d)
            if len(list_dir) > 0:
                if not overright:
                    print("This folder " + d + " is not empty, \nIf you want to save file here, turn overright to True")
                    return

        for c_id in self.unique_id_cells:
            img_cell = csimage.get_label(self.label_image, c_id).astype("uint8")

            # save image and its dilation
            img_cell_dil = ndi.binary_dilation(img_cell, structure=self.struct_dil)
            img_cell_dil = np.subtract(img_cell_dil, img_cell) * 2
            img_comb = np.add(img_cell, img_cell_dil)
            sp_mat = sparse.COO.from_numpy(img_comb)
            sparse.save_npz(os.path.join(self.storage_path + "npz", str(c_id) + ".npz"), sp_mat)

            # # Perform 3D distance map
            # result = ZManalysis.find_cell_axis_center(img_cell, pixel_size)
            # result.to_csv(os.path.join(self.storage_path + "Center_fibre_coords", str(c_id) + ".csv"))

            # save mesh file
            if save_mesh:
                print("This is not implemented yet")
                # ZMimage.make_obj(img_cell, c_id, path+"obj_mesh")

    def cell_segmentation(self):
        """
        Analyse cell parameter such as volume, length, number of neighbor.
        Calculate also plane by plane area, orientation perimeter...
        Parameters
        ----------

        Returns
        -------
        cell_df (pd.DataFrame): result for each cells
        cell_plane_df (pd.DataFrame): result for each plane of each cells
        """

        cell_columns = ["id_im",
                        "vol(um3)",
                        "x_center",
                        "y_center",
                        "z_center",
                        "nb_neighbor",
                        ]
        cell_df = pd.DataFrame(columns=cell_columns)

        cell_plane_columns = ["id_im",
                              "x_center",
                              "y_center",
                              "z_center",
                              ]
        cell_plane_df = pd.DataFrame(columns=cell_plane_columns)

        for c_id in self.unique_id_cells:
            # open files generate by prerequisite
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell_dil = sp_mat.todense()
            img_cell = csimage.get_label(img_cell_dil, 1).astype("uint8")
            # img_cell = img_cell_dil.copy()
            # img_cell[img_cell == 2] = 0
            img_cell_dil[img_cell_dil == 2] = 1

            data_ = csimage.find_cell_axis_center(img_cell, self.pixel_size, resize_image=True)
            # data_ = pd.read_csv(
            #     os.path.join(self.storage_path, "unique_cell/Center_fibre_coords/" + str(c_id) + ".csv"))
            # if "x.micron" in data_.columns:
            #     data_["X.micron"] = data_["x.micron"]

            # measure nb neighbours
            neighbours_id = csimage.find_neighbours_cell_id(img_cell_dil, self.label_image)
            neighbours_id = np.delete(neighbours_id, np.where(neighbours_id == c_id))

            # Get center of the cell
            z, y, x = np.array(np.where(img_cell > 0)).mean(axis=1)

            c_info = pd.DataFrame.from_dict({"id_im": c_id,
                                             "x_center": x,
                                             "y_center": y,
                                             "z_center": z,
                                             "nb_neighbor": len(neighbours_id),
                                             },
                                            orient="index").T

            c_p_info = pd.DataFrame(np.array([np.repeat(c_id, len(data_["x_center"].to_numpy())),
                                              data_["x_center"].to_numpy(),
                                              data_["y_center"].to_numpy(),
                                              data_["z_center"].to_numpy(),
                                              ]).T,
                                    columns=cell_plane_columns)

            cell_df = pd.concat([cell_df, c_info], ignore_index=True)
            cell_plane_df = pd.concat([cell_plane_df, c_p_info], ignore_index=True)

        cell_df.to_csv(os.path.join(self.storage_path, "cell_df.csv"))
        cell_plane_df.to_csv(os.path.join(self.storage_path, "cell_plane_df.csv"))
        return cell_df, cell_plane_df

    def edge_segmentation(self):
        """

        :param cell_df:
        :param cell_plane_df:
        :return:
        """
        cell_df = pd.read_csv(os.path.join(self.storage_path, "cell_df.csv"))
        cell_plane_df = pd.read_csv(os.path.join(self.storage_path, "cell_plane_df.csv"))

        edge_pixel_columns = ["id_im_1",
                              "id_im_2",
                              "id_im_3",
                              "x", "y", "z",
                              "x_cell", "y_cell", "z_cell", ]
        edge_pixel_df = pd.DataFrame(columns=edge_pixel_columns)
        # long axis edge analysis
        edge_columns = ["id_im_1",
                        "id_im_2",
                        "id_im_3",
                        "x_mean", "y_mean", "z_mean",
                        ]
        edge_df = pd.DataFrame(columns=edge_columns)

        for c_id in self.unique_id_cells:

            # step 1
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell1_dil = sp_mat.todense()
            sp_mat = None

            img_cell1_dil[img_cell1_dil == 2] = 1

            neighbours_id = csimage.find_neighbours_cell_id(img_cell1_dil, self.label_image)

            cell_combi = csutils.make_all_list_combination(np.delete(neighbours_id, np.where(c_id == neighbours_id)),
                                                           2)

            # For each edge
            for cb, cc in cell_combi:
                # orient cell counter clockwise
                # need for lateral face analyses
                a_ = csutils.get_angle(
                    (cell_df[cell_df['id_im'] == cb]['x_center'].to_numpy()[0] * self.pixel_size['x_size'],
                     cell_df[cell_df['id_im'] == cb]['y_center'].to_numpy()[0] * self.pixel_size['y_size']),
                    (
                        cell_df[cell_df['id_im'] == c_id]['x_center'].to_numpy()[0] * self.pixel_size[
                            'x_size'],
                        cell_df[cell_df['id_im'] == c_id]['y_center'].to_numpy()[0] * self.pixel_size[
                            'y_size']),
                    (cell_df[cell_df['id_im'] == cc]['x_center'].to_numpy()[0] * self.pixel_size['x_size'],
                     cell_df[cell_df['id_im'] == cc]['y_center'].to_numpy()[0] * self.pixel_size['y_size']))
                if a_ > 0:
                    cc, cb = cb, cc

                sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(int(cb)) + ".npz"))
                img_cell_b_dil = sp_mat.todense()
                img_cell_b_dil[img_cell_b_dil == 2] = 1

                sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(int(cc)) + ".npz"))
                img_cell_c_dil = sp_mat.todense()
                img_cell_c_dil[img_cell_c_dil == 2] = 1

                img_edge = np.multiply(np.multiply(img_cell1_dil, img_cell_b_dil), img_cell_c_dil)

                if len(pd.unique(img_edge.flatten())) < 2:
                    continue

                z0, y0, x0 = np.where(img_edge > 0)

                df = pd.DataFrame({"x": x0 * self.pixel_size['x_size'],
                                   "y": y0 * self.pixel_size['y_size'],
                                   "z": z0 * self.pixel_size['z_size'], })

                df = df.groupby('z').mean()
                df.reset_index(drop=False, inplace=True)

                # relative position to the center of cell
                cell_pos = pd.DataFrame.from_dict({'x': np.round(
                    (cell_plane_df[cell_plane_df['id_im'] == c_id]['x_center'].to_numpy()).astype(float)),
                    'y': np.round((cell_plane_df[cell_plane_df['id_im'] == c_id]['y_center'].to_numpy()).astype(
                        float)),
                    'z': cell_plane_df[cell_plane_df['id_im'] == c_id]['z_center'].to_numpy()})

                x_cell = []
                y_cell = []
                z_cell = []
                for i in range(len(z0)):
                    cell_pos_plane = cell_pos[cell_pos['z'] == z0[i]]
                    if len(cell_pos_plane) != 0:
                        x_cell.append((x0[i] - cell_pos_plane['x']).to_numpy()[0])
                        y_cell.append((y0[i] - cell_pos_plane['y']).to_numpy()[0])
                        z_cell.append(z0[i])

                e_pixel = pd.DataFrame(np.array([np.repeat(c_id, len(x0)),
                                                 np.repeat(cb, len(x0)),
                                                 np.repeat(cc, len(x0)),
                                                 np.array(x0),
                                                 np.array(y0),
                                                 np.array(z0),
                                                 np.array(x_cell),
                                                 np.array(y_cell),
                                                 np.array(z_cell), ]).T,
                                       columns=edge_pixel_columns)
                edge_pixel_df = pd.concat([edge_pixel_df, e_pixel], ignore_index=True)

                tmp = {"id_im_1": c_id,
                       "id_im_2": cb,
                       "id_im_3": cc,
                       "x_mean": x0.mean(),
                       "y_mean": y0.mean(),
                       "z_mean": z0.mean(),
                       }

                c_info = pd.DataFrame.from_dict(tmp,
                                                orient="index").T

                edge_df = pd.concat([edge_df, c_info], ignore_index=True)

        edge_df.to_csv(os.path.join(self.storage_path, "edge_df.csv"))
        edge_pixel_df.to_csv(os.path.join(self.storage_path, "edge_pixel_df.csv"))
        return edge_df, edge_pixel_df

    def face_segmentation(self):
        edge_df = pd.read_csv(os.path.join(self.storage_path, "edge_df.csv"))
        edge_pixel_df = pd.read_csv(os.path.join(self.storage_path, "edge_pixel_df.csv"))

        face_columns = ["id_im_1", "id_im_2",
                        "edge_1", "edge_2",
                        "x", "y", "z",
                        ]
        face_pixel_df = pd.DataFrame(columns=face_columns)
        face_df = pd.DataFrame(columns=["id_im_1", "id_im_2",
                                        "edge_1", "edge_2",
                                        "x_e1_mean", "y_e1_mean", "z_e1_mean",
                                        "x_e2_mean", "y_e2_mean", "z_e2_mean",
                                        "x_mid", "y_mid", "z_mid",
                                       ], )

        for c_id in self.unique_id_cells:


            # open file
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell_dil = sp_mat.todense()
            img_cell_dil[img_cell_dil == 2] = 1

            sub_edges = edge_df[edge_df['id_im_1'] == c_id]

            # cyclic path
            graph, ordered_neighbours, opp_cell = find_ordered_neighbours(sub_edges)
            if len(ordered_neighbours) != 0:

                for c_op_index in opp_cell.keys():
                    c_op, a, b, c, d = opp_cell[c_op_index]
                    sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(int(c_op)) + ".npz"))
                    img_cell_dil2 = sp_mat.todense()
                    img_cell_dil2[img_cell_dil2 == 2] = 1
                    img_face = np.multiply(img_cell_dil, img_cell_dil2)
                    z0, y0, x0 = np.where(img_face > 0)

                    c1 = sub_edges[(sub_edges['id_im_2'] == a) & (sub_edges['id_im_3'] == b)].index[0]
                    c2 = sub_edges[(sub_edges['id_im_2'] == c) & (sub_edges['id_im_3'] == d)].index[0]
                    e1 = pd.DataFrame.from_dict({
                        'x': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                           (edge_pixel_df['id_im_2'] == a) &
                                           (edge_pixel_df['id_im_3'] == b)]['x_cell'].to_numpy(),
                        'y': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                           (edge_pixel_df['id_im_2'] == a) &
                                           (edge_pixel_df['id_im_3'] == b)]['y_cell'].to_numpy(),
                        'z': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                           (edge_pixel_df['id_im_2'] == a) &
                                           (edge_pixel_df['id_im_3'] == b)]['z_cell'].to_numpy()
                    })

                    e2 = pd.DataFrame.from_dict({
                        'x1': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                            (edge_pixel_df['id_im_2'] == c) &
                                            (edge_pixel_df['id_im_3'] == d)]['x_cell'].to_numpy(),
                        'y1': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                            (edge_pixel_df['id_im_2'] == c) &
                                            (edge_pixel_df['id_im_3'] == d)]['y_cell'].to_numpy(),
                        'z': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                           (edge_pixel_df['id_im_2'] == c) &
                                           (edge_pixel_df['id_im_3'] == d)]['z_cell'].to_numpy()
                    })

                    e1_mean = e1.groupby('z').mean()
                    e2_mean = e2.groupby('z').mean()

                    df = (pd.concat((e1_mean, e2_mean), axis=1)).dropna()
                    df['x_mid'] = (df['x'] + df["x1"]) / 2
                    df['y_mid'] = (df['y'] + df["y1"]) / 2


                    f_pixel = pd.DataFrame(np.array([np.repeat(c_id, len(np.array(x0))),
                                                     np.repeat(c_op, len(np.array(x0))),
                                                     np.repeat(c1, len(np.array(x0))),
                                                     np.repeat(c2, len(np.array(x0))),
                                                     np.array(x0),
                                                     np.array(y0),
                                                     np.array(z0),
                                                     ]).T,
                                           columns=face_columns)

                    face_pixel_df = pd.concat([face_pixel_df, f_pixel], ignore_index=True)

                    tmp = {"id_im_1": c_id,
                           "id_im_2": c_op,
                           "edge_1": c1,
                           "edge_2": c2,
                           "x_e1_mean": e1_mean['x'].to_numpy(),
                           "y_e1_mean": e1_mean['y'].to_numpy(),
                           "z_e1_mean": e1_mean.index.to_numpy(),
                           "x_e2_mean": e2_mean['x1'].to_numpy(),
                           "y_e2_mean": e2_mean['y1'].to_numpy(),
                           "z_e2_mean": e2_mean.index.to_numpy(),
                           "x_mid": df['x_mid'].to_numpy(),
                           "y_mid": df['y_mid'].to_numpy(),
                           "z_mid": df.index.to_numpy(),

                           }

                    f_info = pd.DataFrame.from_dict(tmp,
                                                    orient="index").T

                    face_df = pd.concat([face_df, f_info], ignore_index=True)

            # no cyclic path
            graph, paths = find_paths(sub_edges)
            if len(paths) != 0:
                for path_ in paths:
                    for i in range(len(path_)):
                        c_op = path_[i]
                        if len(face_df[(face_df['id_im_1'] == c_id) & (face_df['id_im_2'] == c_op)]) == 0:
                            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(int(c_op)) + ".npz"))
                            img_cell_dil2 = sp_mat.todense()

                            img_cell_dil2[img_cell_dil2 == 2] = 1
                            img_face = np.multiply(img_cell_dil, img_cell_dil2)
                            z0, y0, x0 = np.where(img_face > 0)

                            find_edge_1 = True
                            find_edge_2 = True
                            if i == 0:
                                find_edge_1 = False
                            if i == len(path_) - 1:
                                find_edge_2 = False

                            if find_edge_1:
                                a = path_[i - 1]
                                b = path_[i]
                                c1 = sub_edges[(sub_edges['id_im_2'] == a) & (sub_edges['id_im_3'] == b)].index[0]
                                e1 = pd.DataFrame.from_dict({
                                    'x': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                                       (edge_pixel_df['id_im_2'] == a) &
                                                       (edge_pixel_df['id_im_3'] == b)]['x_cell'].to_numpy(),
                                    'y': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                                       (edge_pixel_df['id_im_2'] == a) &
                                                       (edge_pixel_df['id_im_3'] == b)]['y_cell'].to_numpy(),
                                    'z': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                                       (edge_pixel_df['id_im_2'] == a) &
                                                       (edge_pixel_df['id_im_3'] == b)]['z_cell'].to_numpy()
                                })
                                e1_mean = e1.groupby('z').mean()
                            else:
                                a = np.nan
                                b = np.nan
                                c1 = np.nan
                                e1 = pd.DataFrame.from_dict({'x': [np.nan],
                                                             'y': [np.nan],
                                                             'z': [np.nan]})
                                e1_mean = e1.groupby('z').mean()

                            if find_edge_2:
                                c = path_[i]
                                d = path_[i + 1]

                                c2 = sub_edges[(sub_edges['id_im_2'] == c) & (sub_edges['id_im_3'] == d)].index[0]

                                e2 = pd.DataFrame.from_dict({
                                    'x1': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                                        (edge_pixel_df['id_im_2'] == c) &
                                                        (edge_pixel_df['id_im_3'] == d)]['x_cell'].to_numpy(),
                                    'y1': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                                        (edge_pixel_df['id_im_2'] == c) &
                                                        (edge_pixel_df['id_im_3'] == d)]['y_cell'].to_numpy(),
                                    'z': edge_pixel_df[(edge_pixel_df['id_im_1'] == c_id) &
                                                       (edge_pixel_df['id_im_2'] == c) &
                                                       (edge_pixel_df['id_im_3'] == d)]['z_cell'].to_numpy()
                                })
                                e2_mean = e2.groupby('z').mean()
                            else:
                                c = np.nan
                                d = np.nan
                                c2 = np.nan
                                e2 = np.nan
                                e2 = pd.DataFrame.from_dict({'x1': [np.nan],
                                                             'y2': [np.nan],
                                                             'z': [np.nan]})

                            #             if e1_mean is np.nan:
                            #                 df = e2_mean
                            #                 df['x'] = np.repeat(np.nan, df.shape[0])
                            #                 df['y'] = np.repeat(np.nan, df.shape[0])
                            #             else:
                            df = (pd.concat((e1_mean, e2_mean), axis=1)).dropna()
                            df['x_mid'] = (df['x'] + df["x1"]) / 2
                            df['y_mid'] = (df['y'] + df["y1"]) / 2
                            df['length_um'] = np.sqrt((df['x'] - df['x1']) ** 2.0 + (df['y'] - df['y1']) ** 2.0) * \
                                              self.pixel_size[
                                                  'x_size']
                            df['angle'] = (np.arctan2((df['y1'] - df['y_mid']).to_numpy(),
                                                      (df['x1'] - df['x_mid']).to_numpy()) * 180 / np.pi)

                            f_pixel = pd.DataFrame(np.array([np.repeat(c_id, len(np.array(x0))),
                                                             np.repeat(c_op, len(np.array(x0))),
                                                             np.repeat(c1, len(np.array(x0))),
                                                             np.repeat(c2, len(np.array(x0))),
                                                             np.array(x0),
                                                             np.array(y0),
                                                             np.array(z0),
                                                             ]).T,
                                                   columns=face_columns)

                            face_pixel_df = pd.concat([face_pixel_df, f_pixel], ignore_index=True)

                            tmp = {"id_im_1": c_id,
                                   "id_im_2": c_op,
                                   "edge_1": c1,
                                   "edge_2": c2,
                                   "x_e1_mean": e1_mean['x'].to_numpy(),
                                   "y_e1_mean": e1_mean['y'].to_numpy(),
                                   "z_e1_mean": e1_mean.index.to_numpy(),
                                   "x_e2_mean": e2_mean['x1'].to_numpy(),
                                   "y_e2_mean": e2_mean['y1'].to_numpy(),
                                   "z_e2_mean": e2_mean.index.to_numpy(),
                                   "x_mid": df['x_mid'].to_numpy(),
                                   "y_mid": df['y_mid'].to_numpy(),
                                   "z_mid": df.index.to_numpy(),
                                   "lengths": df['length_um'].to_numpy(),
                                   "angles": df['angle'].to_numpy()
                                   }

                            f_info = pd.DataFrame.from_dict(tmp,
                                                            orient="index").T

                            face_df = pd.concat([face_df, f_info], ignore_index=True)

        face_df.to_csv(os.path.join(self.storage_path, "face_df.csv"))
        face_pixel_df.to_csv(os.path.join(self.storage_path, "face_pixel_df.csv"))



def find_ordered_neighbours(sub_edges):
    """
    Find all cyclic path in neighbours
    Parameters
    ----------
    sub_edges (pd.DataFrame)

    Returns
    -------
    G (Digraph): graph of cell neighbours
    ordered_neighbours (list):
    opp_cell (dict):
    """
    # Create Directed Graph
    G = nx.DiGraph()
    # Add a list of nodes:
    G.add_nodes_from(np.unique(sub_edges[['id_im_2', 'id_im_3']].to_numpy()))
    # Add a list of edges:
    G.add_edges_from(sub_edges[['id_im_2', 'id_im_3']].to_numpy())
    sorted_edges = sorted(nx.simple_cycles(G))

    ordered_neighbours = []
    for i in range(len(sorted_edges)):
        ordered_neighbours.append(np.array([sorted_edges[i],
                                            sorted_edges[i][1:] + [sorted_edges[i][0]]]).flatten().reshape(
            len(sorted_edges[i]), 2, order='F'))
    cpt = 0
    opp_cell = {}
    for i in range(len(ordered_neighbours)):
        couple = ordered_neighbours[i]
        for start, end in zip(couple, np.concatenate((couple[1:], [couple[0]]))):
            if tuple(np.concatenate([[start[1]], start, end])) not in opp_cell.values():
                opp_cell[cpt] = (tuple(np.concatenate([[start[1]], start, end])))
                cpt += 1

    return G, ordered_neighbours, opp_cell


def find_paths(sub_edges):
    """
    Find all non cyclic path in neighbours
    Parameters
    ----------
    sub_edges (pd.DataFrame):

    Returns
    -------

    """
    # Create Directed Graph
    G = nx.DiGraph()
    # Add a list of nodes:
    G.add_nodes_from(np.unique(sub_edges[['id_im_2', 'id_im_3']].to_numpy()))
    # Add a list of edges:
    G.add_edges_from(sub_edges[['id_im_2', 'id_im_3']].to_numpy())

    # Find all paths
    roots = []
    leaves = []
    for node in G.nodes:
        if G.in_degree(node) == 0:  # it's a root
            roots.append(node)
        elif G.out_degree(node) == 0:  # it's a leaf
            leaves.append(node)
    paths = []
    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_paths(G, root, leaf):
                paths.append(path)

    return G, paths


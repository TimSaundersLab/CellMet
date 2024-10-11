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


class Segmentation:

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
            self.pixel_size = dict(x_size=1, y_size=1, z_size=1)
        self.voxel_size = np.prod(list(self.pixel_size.values()))
        self.struct_dil = csutils.generate_struct_dil()

        if path is not None:
            self.storage_path = path
        else:
            self.storage_path = ""

        if nb_core is None:
            self.nb_core = os.cpu_count() - 2
        else:
            self.nb_core = nb_core

    def perform_prerequisite(self, overwrite=False, save_mesh=False, meshtype="ply"):
        """
        Perform prerequisite that is required for further analysis.
        This function :
            - find the center of each cell (in z(depth) axis)
            - save one sparse matrix per cell that contains its pixel and dilated
        Parameters
        ----------
        overwrite (bool): default False, allow to over right if directory "Center_fibre_coord" is not empty
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
                if not overwrite:
                    if (d=="obj_mesh") and save_mesh:
                        print("This folder " + d + " is not empty, \nIf you want to save file here, turn overwrite to True")
        list_dir = os.listdir(self.storage_path + "npz")
        if overwrite or (len(list_dir)==0):
            delayed_call = [
                joblib.delayed(save_unique_cell)(self, c_id)
                for c_id in self.unique_id_cells]
            # joblib.Parallel(n_jobs=self.nb_core, prefer="threads")(delayed_call)
            with csutils.tqdm_joblib(desc="Save NPZ", total=len(self.unique_id_cells)) as progress_bar:
                joblib.Parallel(n_jobs=self.nb_core, prefer="threads")(delayed_call)

        # save mesh file
        if save_mesh:
            # csio.make_mesh_file(self.label_image,
            #                     self.unique_id_cells,
            #                     meshtype=meshtype,
            #                     path=os.path.join(self.storage_path, "obj_mesh"))
            delayed_call = [
                joblib.delayed(csio.make_mesh_file_para)(self.label_image,
                                                           c_id,
                                                           meshtype=meshtype,
                                                           path=os.path.join(self.storage_path, "obj_mesh"))
                for c_id in self.unique_id_cells]
            # joblib.Parallel(n_jobs=self.nb_core)(delayed_call)
            with csutils.tqdm_joblib(desc="Save mesh", total=len(self.unique_id_cells)) as progress_bar:
                joblib.Parallel(n_jobs=self.nb_core, prefer="threads")(delayed_call)

    def all_segmentation(self):
        self.cell_segmentation()
        self.edge_segmentation()
        self.face_segmentation()

    def cell_segmentation_simplified(self):
        cell_columns = ["id_im",
                        "nb_neighbor",
                        "ids_neighbor"
                        ]
        cell_df = pd.DataFrame(columns=cell_columns)
        for c_id in self.unique_id_cells:
            c_id = int(c_id)
            # open image
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell_dil = sp_mat.todense()
            img_cell_dil[img_cell_dil == 2] = 1

            # measure nb neighbours
            neighbours_id, _ = csimage.find_neighbours_cell_id(img_cell_dil, self.label_image, by_plane=False)
            neighbours_id = np.delete(neighbours_id, np.where(c_id == neighbours_id))
            # Populate cell dataframe
            cell_df.loc[len(cell_df)] = {"id_im": int(c_id),
                                         "nb_neighbor": len(neighbours_id),
                                         "ids_neighbor": "".join([str(n) + ';' for n in neighbours_id])
                                         }
            cell_df.to_csv(os.path.join(self.storage_path, "cell_simple_df.csv"))

        # Save dataframe
        cell_df.to_csv(os.path.join(self.storage_path, "cell_simple_df.csv"))

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
                        "z_center",
                        "nb_neighbor",
                        ]
        cell_df = pd.DataFrame(columns=cell_columns)

        cell_plane_columns = ["id_im",
                              "x_center",
                              "y_center",
                              "z_center",
                              "nb_neighbor"
                              ]
        cell_plane_df = pd.DataFrame(columns=cell_plane_columns)

        # for c_id in self.unique_id_cells:
        for i in tqdm(range(len(self.unique_id_cells)), desc="Cell", leave=True):
            c_id = self.unique_id_cells[i]
            c_id = int(c_id)
            # open image
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell_dil = sp_mat.todense()
            img_cell = csimage.get_label(img_cell_dil, 1).astype("uint8")
            img_cell_dil[img_cell_dil == 2] = 1

            data_ = csimage.find_cell_axis_center(img_cell, self.pixel_size, resize_image=True)

            # measure nb neighbours
            neighbours_id, nb_neighbors_plane = csimage.find_neighbours_cell_id(img_cell_dil, self.label_image, by_plane=True)

            # Get center of the cell
            sparce_cell = sparse.COO.from_numpy(img_cell)
            z, y, x = sparce_cell.coords.mean(axis=1)

            # Populate cell dataframe
            cell_df.loc[len(cell_df)] = {"id_im": int(c_id),
                                         "x_center": x,
                                         "y_center": y,
                                         "z_center": z,
                                         "nb_neighbor": len(neighbours_id) - 1,
                                         }
            cell_df.to_csv(os.path.join(self.storage_path, "cell_df.csv"))
            c_p_info = pd.DataFrame(np.array([np.repeat(c_id, len(data_["x_center"].to_numpy())),
                                              data_["x_center"].to_numpy(),
                                              data_["y_center"].to_numpy(),
                                              data_["z_center"].to_numpy(),
                                              nb_neighbors_plane.T,
                                              ]).T,
                                    columns=cell_plane_columns)

            # cell_plane_df = pd.concat([cell_plane_df, c_p_info], ignore_index=True)
            cell_plane_df = pd.concat([df for df in [cell_plane_df, c_p_info] if not df.empty],
                                      ignore_index=True)

            # Save dataframe
            cell_df.to_csv(os.path.join(self.storage_path, "cell_df.csv"))
            cell_plane_df.to_csv(os.path.join(self.storage_path, "cell_plane_df.csv"))

    def edge_segmentation(self):
        """

        :return:
        """

        # Open cell dataframe
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
                        "x_center", "y_center", "z_center",
                        ]
        edge_df = pd.DataFrame(columns=edge_columns)

        # for c_id in self.unique_id_cells:
        for i in tqdm(range(len(self.unique_id_cells)), desc="Cell", leave=True):
            c_id = self.unique_id_cells[i]
            # step 1
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell1_dil = sp_mat.todense()

            img_cell1_dil[img_cell1_dil == 2] = 1

            neighbours_id, _ = csimage.find_neighbours_cell_id(img_cell1_dil, self.label_image, by_plane=False)

            cell_combi = csutils.make_all_list_combination(np.delete(neighbours_id, np.where(c_id == neighbours_id)),
                                                           2)

            delayed_call = [
                joblib.delayed(edge_detection)(self, cell_df, cell_plane_df, edge_pixel_columns, c_id, img_cell1_dil,
                                               cb, cc)
                for cb, cc in cell_combi]
            # res = joblib.Parallel(n_jobs=self.nb_core)(delayed_call)
            with csutils.tqdm_joblib(desc="Edge segmentation", total=len(cell_combi)) as progress_bar:
                res = joblib.Parallel(n_jobs=self.nb_core, prefer="threads")(delayed_call)

            res = [(e_pixel, e_dict) for e_pixel, e_dict in res if e_pixel is not None]
            for e_pixel, e_dict in res:
                edge_pixel_df = pd.concat([df for df in [edge_pixel_df, e_pixel] if not df.empty],
                                          ignore_index=True)
                edge_df.loc[len(edge_df)] = e_dict

            edge_df.to_csv(os.path.join(self.storage_path, "edge_df.csv"))
            edge_pixel_df.to_csv(os.path.join(self.storage_path, "edge_pixel_df.csv"))


    def face_segmentation(self):
        """

        :return:
        """
        edge_df = pd.read_csv(os.path.join(self.storage_path, "edge_df.csv"))
        edge_pixel_df = pd.read_csv(os.path.join(self.storage_path, "edge_pixel_df.csv"))

        face_columns = ["id_im_1", "id_im_2",
                        "edge_1", "edge_2",
                        "x", "y", "z",
                        ]
        face_pixel_df = pd.DataFrame(columns=face_columns)
        face_df = pd.DataFrame(columns=["id_im_1", "id_im_2",
                                        "edge_1", "edge_2",
                                        ], )

        face_edge_pixel_df = pd.DataFrame(columns=["id_im_1", "id_im_2",
                                                   "edge_1", "edge_2",
                                                   "x_e1_mean", "y_e1_mean", "z_e1_mean",
                                                   "x_e2_mean", "y_e2_mean", "z_e2_mean",
                                                   "x_mid", "y_mid", "z_mid", ])
        # for c_id in self.unique_id_cells:
        for i in tqdm(range(len(self.unique_id_cells)), desc="Cell", leave=True):
            c_id = self.unique_id_cells[i]
            # open file
            sp_mat = sparse.load_npz(os.path.join(self.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell_dil = sp_mat.todense()
            img_cell_dil[img_cell_dil == 2] = 1

            sub_edges = edge_df[edge_df['id_im_1'] == c_id]

            ordered_neighbours, opp_cell = find_all_neighbours(sub_edges)

            if len(ordered_neighbours) != 0:
                delayed_call = [
                    joblib.delayed(face_detection)(self.storage_path, c_id, c_op_index, opp_cell, img_cell_dil,
                                                   sub_edges, face_df.columns, face_pixel_df.columns,
                                                   face_edge_pixel_df.columns, edge_pixel_df
                                                   )
                    for c_op_index in opp_cell.keys()]
                # res = joblib.Parallel(n_jobs=self.nb_core)(delayed_call)
                with csutils.tqdm_joblib(desc="Face segmentation", total=len(opp_cell.keys())) as progress_bar:
                    res = joblib.Parallel(n_jobs=self.nb_core, prefer="threads")(delayed_call)

                res = [(f_edge_pixel, f_pixel, f_dict) for f_edge_pixel, f_pixel, f_dict in res if f_pixel is not None]
                for f_edge_pixel, f_pixel, f_dict in res:
                    # face_edge_pixel_df = pd.concat([df for df in [face_edge_pixel_df, f_edge_pixel] if not df.empty],
                    #                                ignore_index=True)
                    face_edge_pixel_df = pd .concat([face_edge_pixel_df, f_edge_pixel], ignore_index=True)
                    face_pixel_df = pd.concat([df for df in [face_pixel_df, f_pixel] if not df.empty],
                                              ignore_index=True)
                    face_df.loc[len(face_df)] = f_dict

            face_df.drop_duplicates(["id_im_1", "id_im_2", "edge_1", "edge_2"], inplace=True)
            face_df.reset_index(drop=True, inplace=True)
            face_df.to_csv(os.path.join(self.storage_path, "face_df.csv"))

            face_edge_pixel_df.dropna(subset=['id_im_1'], inplace=True)
            face_edge_pixel_df.reset_index(drop=True, inplace=True)
            face_edge_pixel_df.to_csv(os.path.join(self.storage_path, "face_edge_pixel_df.csv"))
        face_pixel_df.to_csv(os.path.join(self.storage_path, "face_pixel_df.csv"))


def find_all_neighbours(sub_edges):
    graph, ordered_neighbours1, opp_cell1 = find_cyclic_paths(sub_edges)
    graph, ordered_neighbours2, opp_cell2 = find_non_cyclic_paths(sub_edges)

    k_max = len(opp_cell1.keys())
    for k, v in opp_cell2.items():
        opp_cell1[k + k_max] = v

    ordered_neighbours = ordered_neighbours1 + ordered_neighbours2
    return ordered_neighbours, opp_cell1


def find_cyclic_paths(sub_edges):
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
    d_graph = nx.DiGraph()
    # Add a list of nodes:
    d_graph.add_nodes_from(np.unique(sub_edges[['id_im_2', 'id_im_3']].to_numpy()))
    # Add a list of edges:
    d_graph.add_edges_from(sub_edges[['id_im_2', 'id_im_3']].to_numpy())

    sorted_edges = sorted(nx.simple_cycles(d_graph))
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

    return d_graph, ordered_neighbours, opp_cell


def find_non_cyclic_paths(sub_edges):
    """
    Find all non-cyclic path in neighbours
    Parameters
    ----------
    sub_edges (pd.DataFrame):

    Returns
    -------

    """
    # Create Directed Graph
    d_graph = nx.DiGraph()
    # Add a list of nodes:
    d_graph.add_nodes_from(np.unique(sub_edges[['id_im_2', 'id_im_3']].to_numpy()))
    # Add a list of edges:
    d_graph.add_edges_from(sub_edges[['id_im_2', 'id_im_3']].to_numpy())

    # Find all paths
    roots = []
    leaves = []
    for node in d_graph.nodes:
        if d_graph.in_degree(node) == 0:  # it's a root
            roots.append(node)
        elif d_graph.out_degree(node) == 0:  # it's a leaf
            leaves.append(node)

    ordered_neighbours = []
    opp_cell = {}
    cpt = 0
    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_paths(d_graph, root, leaf):
                ordered_neighbours.append(np.array(path))
                path.insert(0, None)
                path.append(None)
                for i in range(1, len(path) - 1):
                    opp_cell[cpt] = (path[i], path[i - 1], path[i], path[i], path[i + 1])
                    cpt += 1

    return d_graph, ordered_neighbours, opp_cell


def save_unique_cell(seg, c_id):
    img_cell = csimage.get_label(seg.label_image, c_id).astype("uint8")

    # save image and its dilation
    img_cell_dil = ndi.binary_dilation(img_cell, structure=seg.struct_dil)
    img_cell_dil = np.subtract(img_cell_dil, img_cell) * 2
    img_comb = np.add(img_cell, img_cell_dil)
    sp_mat = sparse.COO.from_numpy(img_comb)
    sparse.save_npz(os.path.join(seg.storage_path + "npz", str(c_id) + ".npz"), sp_mat)


def edge_detection(seg, cell_df, cell_plane_df, edge_pixel_columns, c_id, img_cell1_dil, cb, cc):
    sp_mat = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(int(cb)) + ".npz"))
    img_cell_b_dil = sp_mat.todense()
    img_cell_b_dil[img_cell_b_dil == 2] = 1

    sp_mat = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(int(cc)) + ".npz"))
    img_cell_c_dil = sp_mat.todense()
    img_cell_c_dil[img_cell_c_dil == 2] = 1

    img_edge = np.multiply(np.multiply(img_cell1_dil, img_cell_b_dil), img_cell_c_dil)

    if len(pd.unique(img_edge.flatten())) < 2:
        return None, None
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

    sparce_edge = sparse.COO.from_numpy(img_edge)
    z0, y0, x0 = sparce_edge.coords

    # relative position to the center of cell
    cell_pos = pd.DataFrame.from_dict({
        'x': np.round((cell_plane_df[cell_plane_df['id_im'] == c_id]['x_center'].to_numpy()).astype(float)),
        'y': np.round((cell_plane_df[cell_plane_df['id_im'] == c_id]['y_center'].to_numpy()).astype(float)),
        'z': cell_plane_df[cell_plane_df['id_im'] == c_id]['z_center'].to_numpy()
    })

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
                                     np.array(z_cell)]).T,
                           columns=edge_pixel_columns)
    e_dict = {"id_im_1": int(c_id),
              "id_im_2": int(cb),
              "id_im_3": int(cc),
              "x_center": x0.mean(),
              "y_center": y0.mean(),
              "z_center": z0.mean(),
              }
    return e_pixel, e_dict


def face_detection(path, c_id, c_op_index, opp_cell, img_cell_dil, sub_edges, face_columns, face_pixel_columns,
                   face_edge_pixel_columns, edge_pixel_df):
    c_op, a, b, c, d = opp_cell[c_op_index]

    sp_mat = sparse.load_npz(os.path.join(path, "npz/" + str(int(c_op)) + ".npz"))
    img_cell_dil2 = sp_mat.todense()
    img_cell_dil2[img_cell_dil2 == 2] = 1
    img_face = np.multiply(img_cell_dil, img_cell_dil2)

    sparce_face = sparse.COO.from_numpy(img_face)
    z0, y0, x0 = sparce_face.coords

    if a is None:
        # a = np.nan
        # b = np.nan
        c1 = np.nan
        e1 = pd.DataFrame.from_dict({'x': [np.nan],
                                     'y': [np.nan],
                                     'z': [np.nan]})
    else:
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
    if d is None:
        # c = np.nan
        # d = np.nan
        c2 = np.nan
        e2 = pd.DataFrame.from_dict({'x1': [np.nan],
                                     'y1': [np.nan],
                                     'z': [np.nan]})
    else:
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
                           columns=face_pixel_columns)

    # face_pixel_df = pd.concat([face_pixel_df, f_pixel], ignore_index=True)
    # face_pixel_df = pd.concat([df for df in [face_pixel_df, f_pixel] if not df.empty],
    #                          ignore_index=True)

    f_dict = {"id_im_1": c_id,
              "id_im_2": c_op,
              "edge_1": c1,
              "edge_2": c2,
              }

    # f_info = pd.DataFrame.from_dict(tmp,
    #                                orient="index").T

    # face_df = pd.concat([face_df, f_info], ignore_index=True)
    # face_df = pd.concat([df for df in [face_df, f_info] if not df.empty],
    #                    ignore_index=True)

    tmp = {"id_im_1": np.repeat(c_id, len(e1_mean['x'])),
           "id_im_2": np.repeat(c_op, len(e1_mean['x'])),
           "edge_1": np.repeat(c1, len(e1_mean['x'])),
           "edge_2": np.repeat(c2, len(e1_mean['x'])),
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

    f_edge_pixel = pd.DataFrame.from_dict(tmp,
                                          orient="index").T

    # face_edge_pixel_df = pd.concat([face_edge_pixel_df, f_info], ignore_index=True)
    # face_edge_pixel_df = pd.concat([df for df in [face_edge_pixel_df, f_info] if not df.empty],
    #                               ignore_index=True)

    return f_edge_pixel, f_pixel, f_dict


import sparse
from skimage import morphology
def create_skeleton(seg):
    binary_image = np.zeros((seg.label_image.shape))

    for c_id in seg.unique_id_cells:
        sp_mat = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
        img_cell1_dil = sp_mat.todense()

        img_cell1_dil[img_cell1_dil == 2] = 1

        neighbours_id, _ = csimage.find_neighbours_cell_id(img_cell1_dil, seg.label_image, by_plane=False,
                                                   background_value=-1)
        neighbours_id = np.delete(neighbours_id, np.where(c_id == neighbours_id))
        for c_nb in neighbours_id:
            if c_nb !=0:
                sp_mat = sparse.load_npz(os.path.join(seg.storage_path, "npz/" + str(c_nb) + ".npz"))
                img_cell2_dil = sp_mat.todense()
                img_cell2_dil[img_cell2_dil == 2] = 1
                img_edge = np.multiply(img_cell1_dil, img_cell2_dil)
                pos = np.where(img_edge > 0)
                binary_image[pos] = 1
    # binary_image = morphology.skeletonize(binary_image)
    return binary_image

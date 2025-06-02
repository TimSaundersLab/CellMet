import os
import sparse
import joblib

import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from skimage.transform import resize

from sklearn.decomposition import PCA

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
        sparse_cell = sparse.load_npz(
            os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
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
                    "tortuosity",
                    "orient_zy",
                    "orient_xy",
                    "orient_xz",
                    "rho",
                    "theta",
                    "phi",
                    "r1",
                    "r2",
                    "r3",
                    "aspect_ratio",
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
                          "circularity",
                          "nb_neighbor",
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
        delayed_call = [joblib.delayed(sc_analysis_parallel)(seg, int(c_id),
                                                             degree_convert)
                        for c_id in
                        seg.unique_id_cells]
        # res = joblib.Parallel(n_jobs=seg.nb_core)(delayed_call)
        with csutils.tqdm_joblib(desc="Cell analysis", total=len(
                seg.unique_id_cells)) as progress_bar:
            res = joblib.Parallel(n_jobs=seg.nb_core, prefer="threads")(
                delayed_call)

        for cell_out, cell_plane_out in res:
            for k in cell_out.keys():
                cell_df.loc[
                    cell_df[cell_df["id_im"] == cell_out.get("id_im")].index[
                        0], k] = cell_out.get(k)
            cell_plane_df = pd.concat([cell_plane_df,
                                       pd.DataFrame(cell_plane_out,
                                                    columns=cell_plane_columns)],
                                      ignore_index=True)
    else:
        for c_id in seg.unique_id_cells:
            res = sc_analysis_parallel(seg, int(c_id))
            for k in res[0].keys():
                cell_df.loc[
                    cell_df[cell_df["id_im"] == res[0].get("id_im")].index[
                        0], k] = res[0].get(k)
            cell_plane_df = pd.concat([cell_plane_df, pd.DataFrame(res[1],
                                                                   columns=cell_plane_columns)],
                                      ignore_index=True)

    # 1 for a sphere
    cell_df['sphericity'] = (np.pi ** (1 / 3) * (6 * cell_df["volume"]) ** (
            2 / 3)) / cell_df["area"]
    # Save dataframe
    cell_df.to_csv(os.path.join(seg.storage_path, "cell_df.csv"))
    cell_plane_df.to_csv(os.path.join(seg.storage_path, "cell_plane_df.csv"))


def sc_analysis_parallel(seg, c_id, degree_convert=True):
    # open image
    sparse_cell = sparse.load_npz(
        os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
    img_cell_dil = sparse_cell.todense()
    img_cell_dil[img_cell_dil == 2] = 1
    img_cell = csimage.get_label(sparse_cell.todense(), 1).astype("uint8")
    data_ = csimage.find_cell_axis_center_every_z(img_cell, seg.pixel_size,
                                          resize_image=True)
    # measure nb neighbours
    neighbours_id, nb_neighbors_plane = csimage.find_neighbours_cell_id(
        img_cell_dil, seg.label_image, by_plane=True, z_planes=data_[
            "z_center"].to_numpy())
    (a, ox, oy, maj, mi, ar, per, cir) = measure_cell_plane(img_cell,
                                                            seg.pixel_size)

    cell_plane_out = np.array([np.repeat(int(c_id), len(data_[
                                                            "z_center"].to_numpy())),
                               data_["x_center"].to_numpy(),
                               data_["y_center"].to_numpy(),
                               data_["z_center"].to_numpy(),
                               data_["x_center_um"].to_numpy(),
                               data_["y_center_um"].to_numpy(),
                               data_["z_center_um"].to_numpy(),
                               a, ox, oy, maj, mi, ar, per, cir,
                               nb_neighbors_plane.T,
                               ]).T
    start = data_[["x_center_um", "y_center_um", "z_center_um"]].to_numpy()[0]
    end = data_[["x_center_um", "y_center_um", "z_center_um"]].to_numpy()[-1]

    convert = 1
    if degree_convert:
        convert = 180 / np.pi
    orient_zy = np.arctan2((end[2] - start[2]),
                           (end[1] - start[1])) * convert
    orient_xy = np.arctan2((end[0] - start[0]),
                           (end[1] - start[1])) * convert
    orient_xz = np.arctan2((end[0] - start[0]),
                           (end[2] - start[2])) * convert

    # -------------------------------------------------------------
    points_3D = data_[["x_center_um", "y_center_um", "z_center_um"]].to_numpy()
    center = np.mean(points_3D, axis=0)
    points_centered = points_3D - center

    # compute covariance and eigenvalue
    cov_matrix = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues (axes lengths squared) and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    # semi axis lengths
    r1, r2, r3 = np.sqrt(eigenvalues[sorted_indices])

    # Major axis direction
    maj_axis_direction = eigenvectors[:, sorted_indices][:, 0]
    maj_axis_direction /= np.linalg.norm(maj_axis_direction)

    # spherical coordinates
    rho = np.linalg.norm(maj_axis_direction)
    theta = np.arccos(maj_axis_direction[2])
    phi = np.arctan2(maj_axis_direction[1], maj_axis_direction[0])

    if r3 == 0 :
        aspect_ratio = np.nan
    else:
        aspect_ratio = r1 / r3
    elongation = r1 / np.mean([r2, r3])
    ellipticity = (r1 - r3) / r1
    eccentricty = np.sqrt(1-(r3**2/r1**2))
    # -------------------------------------------------------------

    volume = (len(sparse_cell.coords[0]) * seg.voxel_size)
    img_resize = resize(sparse_cell.todense() == 2,
                        (int(sparse_cell.shape[0] * seg.pixel_size["z_size"] /
                             seg.pixel_size["x_size"]),
                         sparse_cell.shape[1],
                         sparse_cell.shape[2]))
    area = np.count_nonzero(img_resize == 1) * seg.pixel_size["x_size"] ** 2

    rd, sd, ci = calculate_lengths_tortuosity(data_, columns=["x_center_um",
                                                              "y_center_um",
                                                              "z_center_um"])
    cell_out = {"id_im": int(c_id),
                "volume": volume,
                "area": area,
                "real_dist": rd,
                "short_dist": sd,
                "tortuosity": ci,
                "orient_zy": orient_zy,
                "orient_xy": orient_xy,
                "orient_xz": orient_xz,
                "rho": rho,
                "theta": theta,
                "phi": phi,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "aspect_ratio": aspect_ratio,
                "elongation": elongation,
                "ellipticity": ellipticity,
                "eccentricty": eccentricty,
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
    edge_pixel_df = pd.read_csv(
        os.path.join(seg.storage_path, "edge_pixel_df.csv"),
        index_col="Unnamed: 0")

    # Calculate edge lengths and tortuosity
    real_dist = []
    short_dist = []
    tort_ind = []
    for id_, [c1, c2, c3] in edge_df[
        ['id_im_1', 'id_im_2', 'id_im_3']].iterrows():
        df_ = edge_pixel_df[(edge_pixel_df['id_im_1'] == c1) &
                            (edge_pixel_df['id_im_2'] == c2) &
                            (edge_pixel_df['id_im_3'] == c3)].copy()
        df_['x'] *= seg.pixel_size["x_size"]
        df_['y'] *= seg.pixel_size["y_size"]
        df_['z'] *= seg.pixel_size["z_size"]
        df_ = df_.groupby('z').mean()
        df_.reset_index(drop=False, inplace=True)
        # Smooth data ?
        from scipy.interpolate import splprep, splev
        # check if there is enough data to smooth
        if len(df_[list("xyz")]) < 3:
            sm_point_df = df_[list("xyz")]
        else:
            if len(df_[list("xyz")]) == 3:
                tck, u = splprep(df_[list("xyz")].to_numpy().T, s=2, k=2)
            else:
                tck, u = splprep(df_[list("xyz")].to_numpy().T, s=2)
            u_fine = np.linspace(0, 1, len(df_))
            smoothed_points = np.column_stack(splev(u_fine, tck))
            sm_point_df = pd.DataFrame(smoothed_points, columns=list("xyz"))
            
        rd, sd, ci = calculate_lengths_tortuosity(sm_point_df, columns=list(
            "xyz"))
        real_dist.append(rd)
        short_dist.append(sd)
        tort_ind.append(ci)
    edge_df["real_dist"] = real_dist
    edge_df["short_dist"] = short_dist
    edge_df["tortuosity"] = tort_ind

    # Calculate edge rotation around cell center
    rotation = []
    for id_, [c1, c2, c3] in edge_df[
        ['id_im_1', 'id_im_2', 'id_im_3']].iterrows():
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
    face_edge_pixel_df = pd.read_csv(
        os.path.join(seg.storage_path, "face_edge_pixel_df.csv"),
        index_col="Unnamed: 0")

    face_edge_pixel_df['length_um'] = np.sqrt(
        (face_edge_pixel_df['x_e1_mean'] - face_edge_pixel_df[
            'x_e2_mean']) ** 2.0 + (
                face_edge_pixel_df['y_e1_mean'] - face_edge_pixel_df[
            'y_e2_mean']) ** 2.0) * \
                                      seg.pixel_size[
                                          'x_size']
    face_edge_pixel_df['angle'] = (np.arctan2(
        (face_edge_pixel_df['y_e2_mean'] * seg.pixel_size['y_size'] -
         face_edge_pixel_df['y_mid'] * seg.pixel_size[
             'y_size']).to_numpy(),
        (face_edge_pixel_df['x_e2_mean'] * seg.pixel_size['x_size'] -
         face_edge_pixel_df['x_mid'] * seg.pixel_size[
             'x_size']).to_numpy()) * 180 / np.pi)

    face_edge_pixel_df.to_csv(
        os.path.join(seg.storage_path, "face_edge_pixel_df.csv"))

    face_df = pd.read_csv(os.path.join(seg.storage_path, "face_df.csv"),
                          index_col="Unnamed: 0")
    face_pixel_df = pd.read_csv(
        os.path.join(seg.storage_path, "face_pixel_df.csv"),
        index_col="Unnamed: 0")

    df = \
        face_edge_pixel_df.groupby(["id_im_1", "id_im_2", "edge_1", "edge_2"])[
            "length_um"].sum() * seg.pixel_size["z_size"]
    df.index = df.index.set_names(["id_im_1", "id_im_2", "edge_1", "edge_2"])
    df = df.reset_index()

    # area and perimeter calculation
    # projection into the best 2D surface
    ff_id = []
    for i, val in df.iterrows():
        f_id = face_df[((face_df["id_im_1"] == val["id_im_1"]) & (
                face_df["id_im_2"] == val["id_im_2"])) |
                       (face_df["id_im_1"] == val["id_im_2"]) & (
                               face_df["id_im_2"] == val[
                           "id_im_1"])].index.to_numpy()

        ff_id.append(f_id)

    ff_id = csutils.remove_duplicate_arrays(ff_id)

    # area and perimeter calculation
    # projection into the best 2D surface
    for f_id in ff_id:
        id_im_1 = face_df.loc[f_id].iloc[
            0]["id_im_1"]
        id_im_2 = face_df.loc[f_id].iloc[0]["id_im_2"]

        points_3D = face_pixel_df[((face_pixel_df["id_im_1"] == id_im_1) & (
                face_pixel_df["id_im_2"] == id_im_2))][
                        list("xyz")].to_numpy() * list(
            seg.pixel_size.values())

        flatness = measure_flatness(points_3D)

        # Project into a 2D surface
        center = points_3D.mean(axis=0)
        r_points = points_3D - center
        _, _, rotation = np.linalg.svd(
            r_points.astype(float), full_matrices=False
        )
        rot_pos = np.dot(r_points, rotation.T)

        edges = alpha_shape(rot_pos[:, :2], alpha=1.5, only_outer=True)
        pp = []
        for k, j in edges:
            pp.append(list(rot_pos[k]))
            pp.append(list(rot_pos[j]))

        pp = order_vertices(pp)
        area, perimeter = polygon_area_perimeter(pp)

        face_df.loc[f_id, 'area'] = area
        face_df.loc[f_id, 'perimeter'] = perimeter
        face_df.loc[f_id, 'flatness'] = flatness

    face_df.to_csv(os.path.join(seg.storage_path, "face_df.csv"))


def calculate_lengths_tortuosity(data, columns, smooth=False):
    """
    Calculate length of the cell (real and shortest), and tortuosity.
    Real distance is the distance between each pixel (we supposed to have one pixel per z plane).
    Shortest distance is the distance between the first and last plane of the cell.
    Tortuosity is calculated as $1-(short_dist/real_dist)$ .

    Parameters
    ----------
    data (pd.DataFrame): dataframe that contains position of the cell center
    columns (list): name of columns that contains position

    Returns
    -------
    real_dist (float)
    short_dist (float)
    tortuosity (float)

    """
    if smooth:
        # Fit a B-spline curve to the 3D points
        tck, u = splprep(data[columns].to_numpy().T, s=4)

        # Generate uniformly spaced smoothed points
        u_fine = np.linspace(0, 1, 100)
        smoothed_points = np.column_stack(splev(u_fine, tck))

        real_dist = np.linalg.norm(np.diff(smoothed_points, axis=0),
                                   axis=1).sum()
        short_dist = np.linalg.norm((smoothed_points[-1] - smoothed_points[0]))
    else:
        real_dist = np.linalg.norm(data.diff()[columns].values, axis=1)[
                    1:].sum()
        short_dist = np.linalg.norm(
            (data.iloc[-1] - data.iloc[0])[columns].values, )

    if (real_dist == 0) or (short_dist == 0):
        tortuosity = np.nan
    else:
        tortuosity = 1 - (short_dist / real_dist)

    return real_dist, short_dist, tortuosity


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
    circularity = []
    cell_in_z_plan = np.where(img_cell.sum(axis=1).sum(axis=1) > 0)[0]

    for z in cell_in_z_plan:
        z = int(z)
        points = np.array(np.where(img_cell[z, :, :] > 0)).flatten().reshape(
            len(np.where(img_cell[z, :, :] > 0)[1]),
            2,
            order='F')
        if len(points) > 10:
            # try:
            hull = ConvexHull(points)

            # Measure cell                                   anisotropy
            # need to center the face at 0, 0
            # otherwise the calculation is wrong
            pts = (points - points.mean(axis=0)) * pixel_size['x_size']
            u, s, vh = np.linalg.svd(pts)
            svd = np.concatenate((s, vh[0, :]))

            s.sort()
            s = s[::-1]

            orientation = svd[2:]
            aa=s[0] / s[1]
            ox=orientation[0]
            oy=orientation[1]
            maj=s[0]
            mi=s[1]
            per=hull.area
            a = hull.volume
            cir=4 * np.pi * aa / per ** 2
            # except:
            #     aa=np.nan
            #     ox = np.nan
            #     oy = np.nan
            #     maj = np.nan
            #     mi = np.nan
            #     per = np.nan
            #     a = np.nan
            #     cir = np.nan

            aniso.append(aa)
            orientation_x.append(ox)
            orientation_y.append(oy)
            major.append(maj)
            minor.append(mi)
            perimeter.append(per)
            area.append(a)
            circularity.append(cir)

        else:
            aniso.append(np.nan)
            orientation_x.append(np.nan)
            orientation_y.append(np.nan)
            major.append(np.nan)
            minor.append(np.nan)
            perimeter.append(np.nan)
            area.append(np.nan)
            circularity.append(np.nan)

    return (aniso, orientation_x, orientation_y, major, minor, area,
            perimeter, circularity)


from scipy.spatial import Delaunay


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j,
                    i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        if (s * (s - a) * (s - b) * (s - c)) < 0:
            area = 0
        else:
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        if area != 0:
            circum_r = a * b * c / (4.0 * area)
            if circum_r < alpha:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)
    return edges


def polygon_area_perimeter(vertices):
    """
    Computes the area of a polygon given its vertices using the Shoelace Theorem.

    :param vertices: List of (x, y) tuples representing polygon vertices in order.
    :return: Absolute area of the polygon.
    """
    x = np.array([p[0] for p in vertices])
    y = np.array([p[1] for p in vertices])

    # Apply Shoelace formula
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    vertices = np.array(vertices)
    # Compute the sum of distances between consecutive vertices
    perimeter = 0.0
    for i in range(len(vertices)):
        next_i = (i + 1) % len(vertices)

        # Calculate the Euclidean distance between the current and next vertex
        dist = np.linalg.norm(vertices[i] - vertices[next_i])
        perimeter += dist

    return area, perimeter


def order_vertices(vertices):
    """
    Orders the vertices around the centroid, ensuring a consistent order
    (either clockwise or counterclockwise).
    """
    vertices = np.array(vertices)
    centroid = np.mean(vertices, axis=0)

    # Compute angles relative to centroid
    angles = np.arctan2(vertices[:, 1] - centroid[1],
                        vertices[:, 0] - centroid[0])

    # Sort vertices based on angle
    ordered_vertices = vertices[np.argsort(angles)]

    return ordered_vertices.tolist()


def measure_flatness(points):
    """
    Measures how flat the 3D sheet is using PCA and the third principal
    component.

    :param points: (N, 3) numpy array of 3D points
    :return: A measure of flatness based on the variance in the third principal component.
    """
    pca = PCA(n_components=3)
    pca.fit(points)

    flatness_measure = pca.explained_variance_[2]

    return flatness_measure

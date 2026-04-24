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

# Define all possible metrics and their calculation functions and associated columns
# This dictionary should be defined at the module level in analysis.py
METRIC_CALCULATORS = {}

def _calculate_volume(sparse_cell, seg):
    return  len(sparse_cell.coords[0]) * seg.voxel_size

def _calculate_area_3d(sparse_cell, seg):
    img_resize = resize(sparse_cell.todense() == 2,
                        (int(sparse_cell.shape[0] * seg.pixel_size["z_size"] / seg.pixel_size["x_size"]),
                         sparse_cell.shape[1],
                         sparse_cell.shape[2]))
    return np.count_nonzero(img_resize == 1) * seg.pixel_size["x_size"] ** 2

def _calculate_tortuosity_metrics(data_):
    rd, sd, ci = calculate_lengths_tortuosity(data_, columns=["x_center_um", "y_center_um", "z_center_um"])
    return {"real_dist": rd, "short_dist": sd, "tortuosity": ci}

def _calculate_orientation_angles(data_, degree_convert):
    if len(data_) < 2: # Need at least two points to define an orientation
        return {"orient_zy": np.nan, "orient_xy": np.nan, "orient_xz": np.nan}
    start = data_[["x_center_um", "y_center_um", "z_center_um"]].to_numpy()[0]
    end = data_[["x_center_um", "y_center_um", "z_center_um"]].to_numpy()[-1]
    convert = 180 / np.pi if degree_convert else 1
    return {
        "orient_zy": np.arctan2((end[2] - start[2]), (end[1] - start[1])) * convert,
        "orient_xy": np.arctan2((end[0] - start[0]), (end[1] - start[1])) * convert,
        "orient_xz": np.arctan2((end[0] - start[0]), (end[2] - start[2])) * convert
    }

def _calculate_pca_shape_metrics(data_):
    points_3D = data_[["x_center_um", "y_center_um", "z_center_um"]].to_numpy()
    if len(points_3D) < 3: # PCA needs at least 3 points for 3D
        return {
            "rho": np.nan, "theta": np.nan, "phi": np.nan,
            "r1": np.nan, "r2": np.nan, "r3": np.nan,
            "aspect_ratio": np.nan, "elongation": np.nan,
            "ellipticity": np.nan, "eccentricty": np.nan
        }
    center = np.mean(points_3D, axis=0)
    points_centered = points_3D - center

    cov_matrix = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    r1, r2, r3 = np.sqrt(eigenvalues[sorted_indices])

    maj_axis_direction = eigenvectors[:, sorted_indices][:, 0]
    maj_axis_direction /= np.linalg.norm(maj_axis_direction)

    rho = np.linalg.norm(maj_axis_direction)
    theta = np.arccos(maj_axis_direction[2])
    phi = np.arctan2(maj_axis_direction[1], maj_axis_direction[0])

    aspect_ratio = r1 / r3 if r3 != 0 else np.nan
    elongation = r1 / np.mean([r2, r3]) if np.mean([r2, r3]) != 0 else np.nan
    ellipticity = (r1 - r3) / r1 if r1 != 0 else np.nan
    eccentricty = np.sqrt(1 - (r3**2 / r1**2)) if r1 != 0 else np.nan

    return {
        "rho": rho, "theta": theta, "phi": phi,
        "r1": r1, "r2": r2, "r3": r3,
        "aspect_ratio": aspect_ratio, "elongation": elongation,
        "ellipticity": ellipticity, "eccentricty": eccentricty
    }

def _get_start_end_coords(data_):
    if len(data_) == 0:
        return {
            'x_start': np.nan, 'y_start': np.nan, 'z_start': np.nan,
            'x_end': np.nan, 'y_end': np.nan, 'z_end': np.nan,
        }
    start = data_[["x_center_um", "y_center_um", "z_center_um"]].to_numpy()[0]
    end = data_[["x_center_um", "y_center_um", "z_center_um"]].to_numpy()[-1]
    return {
        'x_start': start[0], 'y_start': start[1], 'z_start': start[2],
        'x_end': end[0], 'y_end': end[1], 'z_end': end[2],
    }

def _measure_cell_plane(img_cell, seg):
    return measure_cell_plane(img_cell, seg.pixel_size)

def _calculate_face_edge_pixel_metrics(face_edge_pixel_df, pixel_size):
    length_um = np.sqrt(
        (face_edge_pixel_df['x_e1_mean'] - face_edge_pixel_df['x_e2_mean']) ** 2.0 +
        (face_edge_pixel_df['y_e1_mean'] - face_edge_pixel_df['y_e2_mean']) ** 2.0) * \
                pixel_size['x_size']
    angle = (np.arctan2(
        (face_edge_pixel_df['y_e2_mean'] * pixel_size['y_size'] -
         face_edge_pixel_df['y_mid'] * pixel_size['y_size']).to_numpy(),
        (face_edge_pixel_df['x_e2_mean'] * pixel_size['x_size'] -
         face_edge_pixel_df['x_mid'] * pixel_size['x_size']).to_numpy()) * 180 / np.pi)
    return {"length_um": length_um, "angle": angle}

def _calculate_face_geometric_metrics(points_3D):
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
    return {"area": area, "perimeter": perimeter, "flatness": flatness}

def _calculate_edge_tortuosity_for_edge(df_, seg):
    df_ = df_.copy()
    df_['x'] *= seg.pixel_size["x_size"]
    df_['y'] *= seg.pixel_size["y_size"]
    df_['z'] *= seg.pixel_size["z_size"]
    df_ = df_.groupby('z').mean()
    df_.reset_index(drop=False, inplace=True)
    # Check if there is enough point to do the analysis
    if len(df_[list("xyz")]) < 2:
        return np.nan, np.nan, np.nan
    
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

    return calculate_lengths_tortuosity(sm_point_df, columns=list("xyz"))

def _calculate_edge_rotation_for_edge(df_, seg):
    df_ = df_.copy()
    df_['x_cell'] *= seg.pixel_size["x_size"]
    df_['y_cell'] *= seg.pixel_size["y_size"]
    df_['z_cell'] *= seg.pixel_size["z_size"]
    df_.reset_index(drop=False, inplace=True)
    rotation = [0]
    for i in range(1, len(df_)):
        angle_rot = csutils.get_angle(
            (df_["x_cell"][i], df_["y_cell"][i]),
            (0, 0),
            (df_["x_cell"][i - 1], df_["y_cell"][i - 1]))
        rotation.append(angle_rot)
    return rotation


METRIC_CALCULATORS["volume"] = {"type": "cell", "func": _calculate_volume, "args": ["sparse_cell", "seg"], "columns": ["volume"]}
METRIC_CALCULATORS["area_3d"] = {"type": "cell", "func": _calculate_area_3d, "args": ["sparse_cell", "seg"], "columns": ["area"]}
METRIC_CALCULATORS["tortuosity_metrics"] = {"type": "cell", "func": _calculate_tortuosity_metrics, "args": ["data_"], "columns": ["real_dist", "short_dist", "tortuosity"]}
METRIC_CALCULATORS["orientation_angles"] = {"type": "cell", "func": _calculate_orientation_angles, "args": ["data_", "degree_convert"], "columns": ["orient_zy", "orient_xy", "orient_xz"]}
METRIC_CALCULATORS["pca_shape_metrics"] = {"type": "cell", "func": _calculate_pca_shape_metrics, "args": ["data_"], "columns": ["rho", "theta", "phi", "r1", "r2", "r3", "aspect_ratio", "elongation", "ellipticity", "eccentricty"]}
METRIC_CALCULATORS["start_end_coords"] = {"type": "cell", "func": _get_start_end_coords, "args": ["data_"], "columns": ['x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end']}

METRIC_CALCULATORS["sphericity"] = {"type": "post_cell", "depends_on": ["volume", "area_3d"], "func": lambda cell_df: (np.pi ** (1 / 3) * (6 * cell_df["volume"]) ** (2 / 3)) / cell_df["area"], "columns": ["sphericity"]}

METRIC_CALCULATORS["plane_centers"] = {"type": "plane", "func": lambda data_: data_[["x_center", "y_center", "z_center", "x_center_um", "y_center_um", "z_center_um"]], "args": ["data_"], "columns": ["x_center", "y_center", "z_center", "x_center_um", "y_center_um", "z_center_um"]}
METRIC_CALCULATORS["plane_shape_metrics"] = {"type": "plane", "func": _measure_cell_plane, "args": ["img_cell", "seg"], "columns": ["aniso", "orientation_x", "orientation_y", "major", "minor", "area", "perimeter", "circularity"]}
METRIC_CALCULATORS["plane_nb_neighbor"] = {"type": "plane", "func": lambda nb_neighbors_plane: nb_neighbors_plane, "args": ["nb_neighbors_plane"], "columns": ["nb_neighbor"]}

METRIC_CALCULATORS["face_edge_pixel_metrics"] = {"type": "face_edge_pixel", "func": _calculate_face_edge_pixel_metrics, "args": ["face_edge_pixel_df", "pixel_size"], "columns": ["length_um", "angle"]}
METRIC_CALCULATORS["face_geometric_metrics"] = {"type": "face", "func": _calculate_face_geometric_metrics, "args": ["points_3D"], "columns": ["area", "perimeter", "flatness"]}

METRIC_CALCULATORS["edge_tortuosity_metrics"] = {"type": "edge", "func": _calculate_edge_tortuosity_for_edge, "args": ["df_", "seg"], "columns": ["real_dist", "short_dist", "tortuosity"]}
METRIC_CALCULATORS["edge_rotation_metrics"] = {"type": "edge_pixel", "func": _calculate_edge_rotation_for_edge, "args": ["df_", "seg"], "columns": ["rotation"]}


def simplified_cell_analysis(seg):
    """
    Only measure the volume to filter the segmentation
    :param seg:
    :return:
    """
    cell_df = pd.DataFrame(columns=["volume"])
    cell_df["id_im"] = seg.unique_id_cells
    for c_id in seg.unique_id_cells:
        try:
            sparse_cell = sparse.load_npz(
                os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
            img_cell_dil = sparse_cell.todense()
            img_cell_dil[img_cell_dil == 2] = 1
            img_cell = csimage.get_label(sparse_cell.todense(), 1).astype("uint8")
            volume = (len(sparse_cell.coords[0]) * seg.voxel_size)
            cell_df.loc[cell_df[cell_df["id_im"] == c_id].index, "volume"] = volume
            cell_df.to_csv(os.path.join(seg.storage_path, "cell_df.csv"))
        except Exception as e:
            print(f"Error in simplified analysis for cell id {c_id}: {e}")
            continue

def cell_analysis(seg: Segmentation, parallelized=True, degree_convert=True, metrics_to_calculate=None):
    """
    Analyse cell shape that only require one cell.
    :param seg: Segmentation object
    :param parallelized: bool to parallelized analysis
    :param degree_convert: bool to convert angle measure in degree
    :param metrics_to_calculate: list of strings, names of metrics to calculate. If None, all non-post-cell metrics are calculated. Available list of metrics are ["volume", "area_3d", "tortuosity_metrics", "orientation_angles", "pca_shape_metrics]
    """
    if metrics_to_calculate is None:
        metrics_to_calculate = [name for name, info in METRIC_CALCULATORS.items() if info["type"] != "post_cell"]

    # Ensure dependencies for 'post_cell' metrics are included
    for metric_name, metric_info in METRIC_CALCULATORS.items():
        if metric_info["type"] == "post_cell" and metric_name in metrics_to_calculate:
            for dep in metric_info.get("depends_on", []):
                if dep not in metrics_to_calculate:
                    metrics_to_calculate.append(dep)

    # Dynamically build cell_columns and plane_columns
    cell_columns = ["id_im"]
    plane_columns = ["id_im"] # Start with id_im for plane_df

    for metric_name in metrics_to_calculate:
        metric_info = METRIC_CALCULATORS.get(metric_name)
        if metric_info:
            if metric_info["type"] == "cell":
                cell_columns.extend(metric_info["columns"])
            elif metric_info["type"] == "plane":
                plane_columns.extend(metric_info["columns"])
    
    # Add post_cell metric columns if they are requested
    for metric_name in metrics_to_calculate:
        metric_info = METRIC_CALCULATORS.get(metric_name)
        if metric_info and metric_info["type"] == "post_cell":
            cell_columns.extend(metric_info["columns"])

    if os.path.exists(os.path.join(seg.storage_path, "cell_df.csv")):
        cell_df = pd.read_csv(os.path.join(seg.storage_path, "cell_df.csv"),
                              index_col="Unnamed: 0")
        for c in cell_columns:
            if c not in cell_df.columns:
                cell_df[c] = np.nan
    else:
        cell_df = pd.DataFrame(columns=cell_columns)
        cell_df["id_im"] = seg.unique_id_cells

    # Initialize an empty list to collect all cell_plane_out DataFrames
    all_cell_plane_dfs = []

    if parallelized:
        delayed_call = [joblib.delayed(sc_analysis_parallel)(seg, int(c_id), degree_convert, metrics_to_calculate)
                        for c_id in
                        seg.unique_id_cells]
        # res = joblib.Parallel(n_jobs=seg.nb_core)(delayed_call)
        with csutils.tqdm_joblib(desc="Cell analysis", total=len(
                seg.unique_id_cells)) as progress_bar:
            res = joblib.Parallel(n_jobs=seg.nb_core, prefer="threads")(
                delayed_call)
        
        for cell_out, cell_plane_out_df in res:
            # Update cell_df
            idx = cell_df[cell_df["id_im"] == cell_out.get("id_im")].index[0]
            for k, v in cell_out.items():
                if k != "id_im":
                    if type(v) == dict:
                        for key, value in v.items():
                            cell_df.loc[idx, key] = value
                    else:
                        cell_df.loc[idx, k] = v
            # Collect cell_plane_df
            if not cell_plane_out_df.empty:
                all_cell_plane_dfs.append(cell_plane_out_df)
    else:
        for c_id in seg.unique_id_cells:
            cell_out, cell_plane_out_df = sc_analysis_parallel(seg, int(c_id), degree_convert, metrics_to_calculate)
            # Update cell_df
            idx = cell_df[cell_df["id_im"] == cell_out.get("id_im")].index[0]
            for k, v in cell_out.items():
                if k != "id_im":
                    if type(v) == dict:
                        for key, value in v.items():
                            cell_df.loc[idx, key] = value
                    else:
                        cell_df.loc[idx, k] = v
            # Collect cell_plane_df 
            if not cell_plane_out_df.empty:
                all_cell_plane_dfs.append(cell_plane_out_df)

    # Concatenate all collected cell_plane_out DataFrames
    if all_cell_plane_dfs:
        cell_plane_df = pd.concat(all_cell_plane_dfs, ignore_index=True)
    else:
        # If no plane data was generated, create an empty DataFrame with the expected columns
        cell_plane_df = pd.DataFrame(columns=plane_columns)

    # Calculate post_cell metrics (e.g., sphericity)
    if "sphericity" in metrics_to_calculate:
        # Ensure 'volume' and 'area' columns exist before calculating sphericity
        if "volume" in cell_df.columns and "area" in cell_df.columns:
            cell_df['sphericity'] = METRIC_CALCULATORS["sphericity"]["func"](cell_df)
        else:
            print("Warning: 'volume' and 'area' metrics are required for 'sphericity' calculation but were not selected.")
            cell_df['sphericity'] = np.nan

    # Save dataframe
    cell_df.to_csv(os.path.join(seg.storage_path, "cell_df.csv"))
    cell_plane_df.to_csv(os.path.join(seg.storage_path, "cell_plane_df.csv"))


def sc_analysis_parallel(seg, c_id, degree_convert=True, metrics_to_calculate=None):
    try:
        if metrics_to_calculate is None:
            metrics_to_calculate = [name for name, info in METRIC_CALCULATORS.items() if info["type"] != "post_cell"]

        # open image
        sparse_cell = sparse.load_npz(
            os.path.join(seg.storage_path, "npz/" + str(c_id) + ".npz"))
        img_cell_dil = sparse_cell.todense()
        img_cell_dil[img_cell_dil == 2] = 1
        img_cell = csimage.get_label(sparse_cell.todense(), 1).astype("uint8")

        data_ = csimage.find_cell_axis_center_every_z(img_cell, seg.pixel_size,
                                              resize_image=True)
        
        # Handle case where data_ is empty (e.g., cell too small or invalid)
        if data_.empty:
            # Return empty results for this cell
            cell_out = {"id_im": int(c_id)}
            for metric_name in metrics_to_calculate:
                metric_info = METRIC_CALCULATORS.get(metric_name)
                if metric_info and metric_info["type"] == "cell":
                    for col in metric_info["columns"]:
                        cell_out[col] = np.nan
            
            plane_cols = ["id_im"]
            for metric_name in metrics_to_calculate:
                metric_info = METRIC_CALCULATORS.get(metric_name)
                if metric_info and metric_info["type"] == "plane":
                    plane_cols.extend(metric_info["columns"])
            cell_plane_out = pd.DataFrame(columns=plane_cols)
            return cell_out, cell_plane_out

        # measure nb neighbours
        neighbours_id, nb_neighbors_plane = csimage.find_neighbours_cell_id(
            img_cell_dil, seg.label_image, by_plane=True, z_planes=data_[
                "z_center"].to_numpy())
        func_args = {
            "sparse_cell": sparse_cell,
            "seg": seg,
            "img_cell": img_cell,
            "data_": data_,
            "degree_convert": degree_convert,
            "pixel_size": seg.pixel_size,
            "nb_neighbors_plane": nb_neighbors_plane
        }

        cell_out = {"id_im": int(c_id)}
        plane_data_dict = {"id_im": np.repeat(int(c_id), len(data_["z_center"].to_numpy()))}

        for metric_name in metrics_to_calculate:
            metric_info = METRIC_CALCULATORS.get(metric_name)
            if not metric_info:
                continue

            current_args = {arg: func_args.get(arg) for arg in metric_info["args"]}

            if metric_info["type"] == "cell":
                result = metric_info["func"](**current_args)
                cell_out.update({metric_name:result})

            elif metric_info["type"] == "plane":
                result = metric_info["func"](**current_args)
                if metric_name == "plane_centers":
                    for col in metric_info["columns"]:
                        if col in result.columns:
                            plane_data_dict[col] = result[col].to_numpy()
                elif metric_name == "plane_shape_metrics":
                    # result is a tuple of lists (aniso, ox, oy, maj, mi, ar, per, cir)
                    for i, col in enumerate(metric_info["columns"]):
                        plane_data_dict[col] = np.array(result[i])
                elif metric_name == "plane_nb_neighbor":
                    plane_data_dict[metric_info["columns"][0]] = result # result is nb_neighbors_plane array

        # Create cell_plane_out DataFrame
        if plane_data_dict and len(data_["z_center"].to_numpy()) > 0:
            # Ensure all arrays in plane_data_dict have the same length before creating DataFrame
            first_key = next(iter(plane_data_dict))
            expected_len = len(plane_data_dict[first_key])
            cell_plane_out = pd.DataFrame(plane_data_dict)
        else:
            # If no plane metrics were requested or no planes found, create an empty DataFrame with appropriate columns
            plane_cols = ["id_im"]
            for metric_name in metrics_to_calculate:
                metric_info = METRIC_CALCULATORS.get(metric_name)
                if metric_info and metric_info["type"] == "plane":
                    plane_cols.extend(metric_info["columns"])
            cell_plane_out = pd.DataFrame(columns=plane_cols)

        return cell_out, cell_plane_out
    except Exception as e:
        print(f"Error in cell analysis for cell id {c_id}: {e}")
        return {"id_im": int(c_id)}, pd.DataFrame()


def edge_analysis(seg: Segmentation, metrics_to_calculate=None):
    """
    :param seg : Segmentation object
    :param metrics_to_calculate: list of strings, names of metrics to calculate. 
           If None, all edge metrics are calculated. Available: ["edge_tortuosity_metrics", "edge_rotation_metrics"]
    """
    if metrics_to_calculate is None:
        metrics_to_calculate = ["edge_tortuosity_metrics", "edge_rotation_metrics"]

    edge_df_path = os.path.join(seg.storage_path, "edge_df.csv")
    edge_pixel_df_path = os.path.join(seg.storage_path, "edge_pixel_df.csv")

    if not os.path.exists(edge_df_path):
        raise FileNotFoundError(f"Required file not found: {edge_df_path}. Please ensure edge segmentation has been performed.")
    if not os.path.exists(edge_pixel_df_path):
        raise FileNotFoundError(f"Required file not found: {edge_pixel_df_path}. Please ensure edge segmentation has been performed.")

    edge_df = pd.read_csv(edge_df_path, index_col="Unnamed: 0")
    edge_pixel_df = pd.read_csv(edge_pixel_df_path, index_col="Unnamed: 0")

    # Calculate edge lengths and tortuosity
    if "edge_tortuosity_metrics" in metrics_to_calculate:
        real_dist, short_dist, tort_ind = [], [], []
        for id_, [c1, c2, c3] in edge_df[['id_im_1', 'id_im_2', 'id_im_3']].iterrows():
            try:
                df_ = edge_pixel_df[(edge_pixel_df['id_im_1'] == c1) &
                                    (edge_pixel_df['id_im_2'] == c2) &
                                    (edge_pixel_df['id_im_3'] == c3)]
                rd, sd, ci = _calculate_edge_tortuosity_for_edge(df_, seg)
            except Exception as e:
                print(f"Error in edge tortuosity analysis for edge {c1}-{c2}-{c3}: {e}")
                rd, sd, ci = np.nan, np.nan, np.nan
            real_dist.append(rd)
            short_dist.append(sd)
            tort_ind.append(ci)
        edge_df["real_dist"] = real_dist
        edge_df["short_dist"] = short_dist
        edge_df["tortuosity"] = tort_ind

    # Calculate edge rotation around cell center
    if "edge_rotation_metrics" in metrics_to_calculate:
        rotation = []
        for id_, [c1, c2, c3] in edge_df[['id_im_1', 'id_im_2', 'id_im_3']].iterrows():
            df_ = edge_pixel_df[(edge_pixel_df['id_im_1'] == c1) &
                                (edge_pixel_df['id_im_2'] == c2) &
                                (edge_pixel_df['id_im_3'] == c3)]
            try:
                rotation.extend(_calculate_edge_rotation_for_edge(df_, seg))
            except Exception as e:
                print(f"Error in edge rotation analysis for edge {c1}-{c2}-{c3}: {e}")
                rotation.extend([np.nan] * len(df_))
        edge_pixel_df["rotation"] = rotation

    edge_df.to_csv(edge_df_path)
    edge_pixel_df.to_csv(edge_pixel_df_path)


def face_analysis(seg: Segmentation, metrics_to_calculate=None):
    """
    Analyse face parameters.
    :param seg : Segmentation object
    :param metrics_to_calculate: list of strings, names of metrics to calculate. 
           If None, all face metrics are calculated. Available: ["face_edge_pixel_metrics", "face_geometric_metrics"]
    """
    if metrics_to_calculate is None:
        metrics_to_calculate = ["face_edge_pixel_metrics", "face_geometric_metrics"]

    # --- face_edge_pixel_df metrics ---
    if "face_edge_pixel_metrics" in metrics_to_calculate:
        path = os.path.join(seg.storage_path, "face_edge_pixel_df.csv")
        if os.path.exists(path):
            face_edge_pixel_df = pd.read_csv(path, index_col="Unnamed: 0")
            res = _calculate_face_edge_pixel_metrics(face_edge_pixel_df, seg.pixel_size)
            face_edge_pixel_df['length_um'] = res["length_um"]
            face_edge_pixel_df['angle'] = res["angle"]
            face_edge_pixel_df.to_csv(path)

    # --- face_df metrics ---
    if "face_geometric_metrics" in metrics_to_calculate:
        face_df_path = os.path.join(seg.storage_path, "face_df.csv")
        face_pixel_df_path = os.path.join(seg.storage_path, "face_pixel_df.csv")
        face_edge_pixel_df_path = os.path.join(seg.storage_path, "face_edge_pixel_df.csv")
        
        if not os.path.exists(face_df_path):
            raise FileNotFoundError(f"Required file not found: {face_df_path}. Please ensure face segmentation has been performed.")
        if not os.path.exists(face_pixel_df_path):
            raise FileNotFoundError(f"Required file not found: {face_pixel_df_path}. Please ensure face segmentation has been performed.")

        if os.path.exists(face_df_path) and os.path.exists(face_pixel_df_path):
            face_df = pd.read_csv(face_df_path, index_col="Unnamed: 0")
            face_pixel_df = pd.read_csv(face_pixel_df_path, index_col="Unnamed: 0")

            # Determine which faces to process based on edge pixel data
            if os.path.exists(face_edge_pixel_df_path):
                face_edge_pixel_df = pd.read_csv(face_edge_pixel_df_path, index_col="Unnamed: 0")
                
                # Ensure length_um exists for grouping
                if 'length_um' not in face_edge_pixel_df.columns:
                    res = _calculate_face_edge_pixel_metrics(face_edge_pixel_df, seg.pixel_size)
                    face_edge_pixel_df['length_um'] = res["length_um"]
                
                df = face_edge_pixel_df.groupby(["id_im_1", "id_im_2", "edge_1", "edge_2"])["length_um"].sum() * seg.pixel_size["z_size"]
                df.index = df.index.set_names(["id_im_1", "id_im_2", "edge_1", "edge_2"])
                df = df.reset_index()

                ff_id = []
                for i, val in df.iterrows():
                    f_id = face_df[((face_df["id_im_1"] == val["id_im_1"]) & (face_df["id_im_2"] == val["id_im_2"])) |
                                   (face_df["id_im_1"] == val["id_im_2"]) & (face_df["id_im_2"] == val["id_im_1"])].index.to_numpy()
                    ff_id.append(f_id)
                ff_id = csutils.remove_duplicate_arrays(ff_id)

                for f_id in ff_id:
                    try:
                        id_im_1 = face_df.loc[f_id].iloc[0]["id_im_1"]
                        id_im_2 = face_df.loc[f_id].iloc[0]["id_im_2"]

                        points_3D = face_pixel_df[((face_pixel_df["id_im_1"] == id_im_1) & (
                                face_pixel_df["id_im_2"] == id_im_2))][
                                        list("xyz")].to_numpy() * list(seg.pixel_size.values())

                        results = _calculate_face_geometric_metrics(points_3D)
                        face_df.loc[f_id, 'area'] = results['area']
                        face_df.loc[f_id, 'perimeter'] = results['perimeter']
                        face_df.loc[f_id, 'flatness'] = results['flatness']
                    except Exception as e:
                        print(f"Error in face geometric analysis for face id {f_id}: {e}")

                face_df.to_csv(face_df_path)


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
    perimeter (float): in µm

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

from vtk import (vtkPolyData,
                 vtkCellArray,
                 vtkPoints,
                 vtkPolygon,
                 vtkPLYWriter,
                 vtkDecimatePro,
                 vtkSmoothPolyDataFilter,
                 vtkPolyDataNormals,
                 vtkOBJWriter
                 )
import os
from tifffile import tifffile
from skimage import measure
import numpy as np
from . import image as csimage


def array2mesh(vert, face):
    """ Code inspired by https://github.com/selaux/numpy2vtk
    """
    # Handle the points & vertices:
    z_index = 0
    vtk_points = vtkPoints()
    for p in vert:
        z_value = p[2] if vert.shape[1] == 3 else z_index
        vtk_points.InsertNextPoint([p[0], p[1], z_value])
    number_of_points = vtk_points.GetNumberOfPoints()

    indices = np.array(range(number_of_points), dtype=np.int8)
    vtk_vertices = vtkCellArray()
    for v in indices:
        vtk_vertices.InsertNextCell(1)
        vtk_vertices.InsertCellPoint(v)

    # Handle faces
    number_of_polygons = face.shape[0]
    poly_shape = face.shape[1]
    vtk_polygons = vtkCellArray()
    for j in range(0, number_of_polygons):
        polygon = vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(poly_shape)
        for i in range(0, poly_shape):
            polygon.GetPointIds().SetId(i, face[j, i])
        vtk_polygons.InsertNextCell(polygon)

    # Assemble the vtkPolyData from the points, vertices and faces
    poly_data = vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetVerts(vtk_vertices)
    poly_data.SetPolys(vtk_polygons)

    return poly_data


def _get_largest_cc(segmentation):
    """Legacy version, substitute by getLargestCC
    Returns largest connected component """
    """Returns largest connected components"""
    # ~2x faster than clean_object(obj)
    # relabel connected components
    labels = measure.label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largest_cc


def label2vtk(segmentation, label, step_size=1):
    """Compute a mesh from a single object"""
    # Retrieve the segmentation corresponding to the label, keep it if > min_volume
    obj = csimage.get_label(segmentation, label)

    if obj is None:
        return None

    if not np.any(obj):
        # If no index match nothing to do
        return None

    # Get the largest connected component
    obj = _get_largest_cc(obj)
    if obj is None:
        return None

    # cast obj in float a required by marching cube
    obj = obj.astype(float)

    # Generate a mesh using the marching cubes algorithm.
    vertx, faces, normals, _ = measure.marching_cubes(obj, 0, step_size=step_size)
    # Convert the vertices and faces in a VTK polyData
    vtk_poly = array2mesh(vertx, faces.astype(int))
    return vtk_poly

def make_mesh_file_para(image, cell_id, meshtype="ply", step_size=1, path=""):
    vtk_poly = label2vtk(image, cell_id, step_size)
    write_mesh(vtk_poly, meshtype, os.path.join(path, str(cell_id)))

def make_mesh_file(image, id_unique_cell=None, meshtype="ply", step_size=1, path=""):
    """
    Generate and save all mesh file.

    Parameters
    ----------
    image (np.array): labeled image of cells
    id_unique_cell (np.array): array of unique label in the image
    meshtype (str): choose format of mesh file. Can be "ply" or "obj"
    step_size (int): step size for the marching cube algorithm
    path (str): path where to save mesh files


    """
    if id_unique_cell is None:
        id_unique_cell = csimage.get_unique_id_in_image(image)

    for i in id_unique_cell:
        vtk_poly = label2vtk(image, i, step_size)
        write_mesh(vtk_poly, meshtype, os.path.join(path, str(i)))


def write_mesh(vtk_poly, meshtype='ply', savepath=None):
    """
    meshtype can be 'ply' or 'obj'
    """
    if meshtype == "ply":
        writer = vtkPLYWriter()
    elif meshtype == "obj":
        writer = vtkOBJWriter()
    else:
        print("meshtype is wrong. Choose between ply and obj")
        return

    writer.SetInputData(vtk_poly)
    writer.SetFileName(savepath + '.' + meshtype)
    writer.Write()


def write_tiff(image, filename, path="", pixel_size=None, _type="uint8"):
    """
    Save numpy array as tiff file

    Parameters
    ----------
    image (np.array): image to save
    filename (str): name of the image
    path (str): path where to save the image
    pixel_size (dict): dictionary that contains pixel size
    (keys inside the dictionary need to be 'x_size', 'y_size' and 'z_size').
    type (str): format data depth in 8 bits.
    """

    if pixel_size is None:
        pixel_size = dict(x_size=1, y_size=1, z_size=1)
    if len(image.shape) == 3:
        tifffile.imwrite(os.path.join(path, filename + ".tif"),
                         image.astype(_type),
                         imagej=True,
                         resolution=(1 / pixel_size['y_size'], 1 / pixel_size['x_size']),
                         metadata={
                             'spacing': pixel_size['z_size'],
                             'unit': 'um',
                             'axes': 'ZYX',
                             'hyperstack': True,
                             'channels': 1,
                             'slices': image.shape[0]
                         })

    elif len(image.shape) == 2:
        tifffile.imwrite(os.path.join(path, filename + ".tif"),
                         image.astype(_type),
                         imagej=True,
                         resolution=(1 / pixel_size['y_size'], 1 / pixel_size['x_size']),
                         metadata={
                             'unit': 'um',
                             'axes': 'YX',
                             'hyperstack': True,
                             'channels': 1,
                         })


import re
import io

import contextlib

import joblib
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import scipy.ndimage as ndi


@contextlib.contextmanager
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
def tqdm_joblib(*args, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar
    given as argument"""

    tqdm_object = tqdm(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def generate_struct_dil(dim=3):
    struct_dil = ndi.generate_binary_structure(3, 2)
    struct_dil[0] = np.repeat(False, 9).reshape(3, 3)
    struct_dil[2] = np.repeat(False, 9).reshape(3, 3)
    if dim == 2:
        struct_dil = struct_dil[1]

    return struct_dil

def make_all_list_combination(l_value, n):
    """
    Make all combinations of groups of k elements in list

    Parameters
    ----------
    l_value (list): list of elements
    n (int): size of the group for the combination

    Returns
    -------
    l_combination (list): list of combinations
    """
    l_combination = []
    i, imax = 0, 2 ** len(l_value) - 1
    while i <= imax:
        s = []
        j, jmax = 0, len(l_value) - 1
        while j <= jmax:
            if (i >> j) & 1 == 1:
                s.append(l_value[j])
            j += 1
        if len(s) == n:
            s.sort()
            l_combination.append(s)
        i += 1
    return l_combination

def get_angle(a, b, c, degree_convert=True):
    """
    Get angle between the three points A, B and C; $\widehat{ABC}$.

    Parameters
    ----------
    a,b,c (tuples): Point's coordinate
    degree_convert (bool): default True, convert angle in degree

    Returns
    -------
    ang (float): angle between points A, B and C.
    """
    ang = (np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0]))
    ang = ang - (np.pi*2) if ang > np.pi else ang
    ang = (np.pi*2) + ang if ang < -np.pi else ang
    if degree_convert:
        ang = ang / np.pi * 180
    return ang


def str_to_array(string):
    """
    Convert string into array when read csv file.

    Parameters
    ----------
    string (str): string of float value

    Returns
    -------
    list (list): list of float values
    """
    return np.array([float(re.sub('[^.\-\d]', '', x))
                     if 'e-' not in x
                     else float(re.sub('[^.\-\d]', '', x.split('e-')[0]))*float(re.sub('[^.\-\d]', '', x.split('e-')[-1]))
                     for x in np.array(str(string).split(" "))
                     if ((x != "...") and (x != "") and (re.sub('[^.\-\d]', '', x) !=""))])


def reduce_label_size(image):
    """
    Reduce label values in order to reduce pixel size and the size of the image.

    Parameters
    ----------
    image: np.array

    Returns
    -------
    image: np.array with the new label
    bits_size: depth of the pixel size
    """

    unique_label = np.unique(image)

    if len(unique_label) < 2 ** 8:
        bits_size = np.uint8
    elif len(unique_label) < 2 ** 16:
        bits_size = np.uint16
    else:
        bits_size = np.uint32

    cpt = 0
    for ul in unique_label:
        if ul != 0:
            image = np.putmask(np.array(image), image == ul, cpt)
        cpt += 1

    return image, bits_size
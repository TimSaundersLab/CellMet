import re
import numpy as np
import pandas as pd

import scipy.ndimage as ndi


def generate_struct_dil():
    struct_dil = ndi.generate_binary_structure(3, 2)
    struct_dil[0] = np.repeat(False, 9).reshape(3, 3)
    struct_dil[2] = np.repeat(False, 9).reshape(3, 3)
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

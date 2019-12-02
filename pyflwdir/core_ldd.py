# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np

from pyflwdir import core

# LDD type
_ds = np.array([
    [7, 8, 9],
    [4, 5, 6],
    [1, 2, 3]], dtype=np.uint8)
_us = np.array([
    [3, 2, 1],
    [6, 5, 4],
    [9, 8, 7]], dtype=np.uint8)
_mv = np.uint8(255)
_pv = np.uint8(0)
_ldd_ = np.unique(np.concatenate([[_pv], [_mv], _ds.flatten()])) 


def _is_ldd(flwdir):
    return np.all([v in _ldd_ for v in np.unique(flwdir)])

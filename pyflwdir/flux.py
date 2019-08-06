# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# local
# import flow direction definition
from .core import fd
_nodata = fd._nodata
_pits = fd._pits 
_ds = fd._ds

@njit
def propagate_upstream(rnodes, rnodes_up, material):
    shape = material.shape
    material = material.flatten()
    for i in range(len(rnodes)):
        nn_ds = rnodes[-i-1]
        nn_us = rnodes_up[-i-1]
        for j in range(len(nn_ds)):
            idx_ds = nn_ds[j]
            idxs_us = nn_us[j,:] # NOTE: has nodata (-1) values
            v = material[idx_ds]
            for idx_us in idxs_us:
                if idx_us == -1: break
                material[idx_us] += v
    return material.reshape(shape)

@njit
def propagate_downstream(rnodes, rnodes_up, material):
    shape = material.shape
    material = material.flatten()
    for i in range(len(rnodes)):
        for j in range(len(rnodes[i])):
            idx_ds = rnodes[i][j]
            idxs_us = rnodes_up[i][j] # NOTE: has nodata (-1) values
            v = 0
            for idx_us in idxs_us:
                if idx_us == -1: break
                v += material[idx_us]
            material[idx_ds] += v
    return material.reshape(shape)
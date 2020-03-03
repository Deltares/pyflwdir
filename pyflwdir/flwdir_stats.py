# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# Feb 2020

from numba import njit
import numpy as np

# import flow direction definition
from pyflwdir import core
from pyflwdir import numba_stats as ns

_mv = core._mv


@njit
def moving_average(
    data, weights, n, idxs_ds, idxs_us, uparea_sparse, upa_min=0.0, nodata=-9999.0
):
    """Take the moving weighted average over the flow direction network
    
    Parameters
    ----------
    data : 1D (sparse) array
        values to be averaged
    weights : 1D (sparse) array
        weights 
    n : int
        number of up/downstream neighbors to include
    idxs_ds, idxs_us : array of int
        indices of down- and upstream neighbors
    uparea_sparse : 1D array of float
        upstream area
    upa_min : float, optional
        Minimum upstream area for upstream neighbors to be considered
        (by default 0.0)
    nodata : float, optional
        Nodata values which is ignored when calculating the average
        (by default -9999.0)
    
    Returns
    -------
    1D array
        averaged data
    """
    if data.size != idxs_ds.size:
        raise ValueError("data has invalid size")
    data_out = np.full(data.size, nodata, dtype=data.dtype)
    for idx0 in range(data.size):
        if data[idx0] == nodata:
            continue
        idxs = core.flwdir_window(
            idx0, n, idxs_ds, idxs_us, uparea_sparse, upa_min=upa_min
        )
        idxs = idxs[idxs != _mv]
        if idxs.size > 0:
            data_out[idx0] = ns._average(data[idxs], weights[idxs], nodata)
    return data_out

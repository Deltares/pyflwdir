# -*- coding: utf-8 -*-
""""""

from numba import njit
import numpy as np

# import flow direction definition
from pyflwdir import core

_mv = core._mv
__all__ = []

# NOTE np.average (with) weights is not yet supoorted by numpy
# all functions are faster than numpy.
@njit
def _average(data, weights, nodata):
    """Weighted arithmetic mean
    NOTE: does not work with nodata=np.nan!
    """
    v = 0.0
    w = 0.0
    for i in range(data.size):
        v0 = data[i]
        if v0 == nodata:
            continue
        w0 = weights[i]
        v += w0 * v0
        w += w0
    return v / w if w != 0 else nodata


@njit
def _mean(data, nodata):
    """Arithmetic mean
    NOTE: does not work with nodata=np.nan!
    """
    v = 0.0
    w = 0.0
    for v0 in data:
        if v0 == nodata:
            continue
        v += v0
        w += 1.0
    return v / w if w != 0 else nodata


@njit
def moving_average(data, weights, n, idxs_ds, idxs_us_main, nodata=-9999.0, mv=_mv):
    """Take the moving weighted average over the flow direction network.

    Parameters
    ----------
    data : 1D (sparse) array
        values to be averaged
    weights : 1D (sparse) array
        weights
    n : int
        number of up/downstream neighbors to include
    idxs_ds, idxs_us_main : array of int
        indices of downstream, main upstream cells
    upa_min : float, optional
        Minimum upstream area for upstream neighbors to be considered, by default 0.0
    nodata : float, optional
        Nodata value which is ignored when calculating the average, by default -9999.0

    Returns
    -------
    1D array
        averaged data
    """
    # loop over values and avarage
    data_out = np.full(data.size, nodata, dtype=data.dtype)
    for idx0 in range(data.size):
        if data[idx0] == nodata:
            continue
        idxs = core._window(idx0, n, idxs_ds, idxs_us_main)
        idxs = idxs[idxs != mv]
        if idxs.size > 0:
            data_out[idx0] = _average(data[idxs], weights[idxs], nodata)
    return data_out


@njit
def upstream_sum(idxs_ds, data, nodata=-9999.0, mv=_mv):
    """Returns sum of first upstream values

    Parameters
    ----------
    idxs_ds : 1D-array of uint32
        sparse indices of downstream cells

    Returns
    -------
    2D-array of uint32
        indices of upstream cells
    """
    # 2D arrays of upstream index
    arr_sum = np.full(data.size, 0, dtype=data.dtype)
    for idx0 in range(data.size):
        idx_ds = idxs_ds[idx0]
        if idx_ds != mv and idx_ds != idx0:
            if data[idx0] == nodata or data[idx_ds] == nodata:
                arr_sum[idx0] = nodata
            else:
                arr_sum[idx_ds] += data[idx0]
    return arr_sum

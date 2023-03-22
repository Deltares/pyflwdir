# -*- coding: utf-8 -*-
""""""

from numba import njit
import numpy as np

# import flow direction definition
from . import core

_mv = core._mv
__all__ = []


# NOTE np.average (with) weights is not yet supoorted by numpy
# all functions are faster than numpy.
@njit
def _average(data, weights, nodata):
    """Weighted arithmetic mean"""
    v = 0.0
    w = 0.0
    nan = np.isnan(nodata)
    for i in range(data.size):
        v0 = data[i]
        if (not nan and v0 == nodata) or (nan and np.isnan(v0)):
            continue
        w0 = weights[i]
        v += w0 * v0
        w += w0
    return v / w if w != 0 else nodata


@njit
def _mean(data, nodata):
    """Arithmetic mean"""
    v = 0.0
    w = 0.0
    nan = np.isnan(nodata)
    for v0 in data:
        if (not nan and v0 == nodata) or (nan and np.isnan(v0)):
            continue
        v += v0
        w += 1.0
    return v / w if w != 0 else nodata


@njit
def lstsq(x: np.ndarray, y: np.ndarray):
    """Simple ordinary Least Squares regression."""
    n = x.size
    x_sum = 0.0
    y_sum = 0.0
    x_sq_sum = 0.0
    x_y_sum = 0.0

    for i in range(n):
        x_sum += x[i]
        y_sum += y[i]
        x_sq_sum += x[i] ** 2
        x_y_sum += x[i] * y[i]

    slope = (n * x_y_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum**2)
    intercept = (y_sum - slope * x_sum) / n

    return slope, intercept


@njit
def moving_average(
    data, weights, n, idxs_ds, idxs_us_main, strord=None, nodata=-9999.0, mv=_mv
):
    """Take the moving weighted average over the flow direction network.

    Parameters
    ----------
    data : 1D array
        values to be averaged
    weights : 1D array
        weights
    n : int
        number of up/downstream neighbors to include
    idxs_ds, idxs_us_main : array of int
        indices of downstream, main upstream cells
    strord: 1D array
        stream order, when set limit window to cells of same or smaller stream order.
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
        idxs = core._window(idx0, n, idxs_ds, idxs_us_main, strord=strord, mv=mv)
        idxs = idxs[idxs != mv]
        if idxs.size > 0:
            w = np.ones(idxs.size) if weights is None else weights[idxs]
            data_out[idx0] = _average(data[idxs], w, nodata)
    return data_out


@njit
def moving_median(data, n, idxs_ds, idxs_us_main, strord=None, nodata=-9999.0, mv=_mv):
    """Take the moving median over the flow direction network.

    Parameters
    ----------
    data : 1D (sparse) array
        values
    weights : 1D (sparse) array
        weights
    n : int
        number of up/downstream neighbors to include
    idxs_ds, idxs_us_main : array of int
        indices of downstream, main upstream cells
    strord: 1D array
        stream order, when set limit window to cells of same or smaller stream order.
    nodata : float, optional
        Nodata value which is ignored when calculating the median, by default -9999.0

    Returns
    -------
    1D array
        median data
    """
    # loop over values and avarage
    data_out = np.full(data.size, nodata, dtype=data.dtype)
    nan = np.isnan(nodata)
    for idx0 in range(data.size):
        if data[idx0] == nodata:
            continue
        idxs = core._window(idx0, n, idxs_ds, idxs_us_main, strord=strord, mv=mv)
        idxs = idxs[idxs != mv]
        if idxs.size > 0:
            a = data[idxs]
            if not nan:
                a = np.where(a == nodata, np.nan, a).astype(a.dtype)
            data_out[idx0] = np.nanmedian(a)
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

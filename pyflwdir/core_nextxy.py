# -*- coding: utf-8 -*-
"""Description of NEXTXY flow direction type and methods to convert to/from general
nextidx. This type is mainly used for the CaMa-Flood model. Note that X (column) and Y
(row) coordinates are one-based."""

from pathlib import Path
from typing import List, Union
from numba import njit
import numpy as np
from . import core, gis_utils

__all__ = ["read_nextxy"]

# NEXTXY type
_ftype = "nextxy"
_mv = np.int32(-9999)
# -10 is inland termination, -9 river outlet at ocean
_pv = np.array([-9, -10], dtype=np.int32)
# NOTE: data below for consistency with LDD / D8 types and testing
_us = np.ones((2, 3, 3), dtype=np.int32) * 2
_us[:, 1, 1] = _pv[0]


def from_array(flwdir, dtype=np.intp):
    if not (
        (isinstance(flwdir, tuple) and len(flwdir) == 2)
        or (
            isinstance(flwdir, np.ndarray) and flwdir.ndim == 3 and flwdir.shape[0] == 2
        )
    ):
        raise TypeError("NEXTXY flwdir data not understood")
    nextx, nexty = flwdir  # convert [2,:,:] OR ([:,:], [:,:]) to [:,:], [:,:]
    return _from_array(nextx, nexty, dtype=dtype)


def to_array(idxs_ds, shape, mv=core._mv):
    nextx, nexty = _to_array(idxs_ds, shape, mv=mv)
    return np.stack([nextx, nexty])


@njit
def _from_array(nextx, nexty, _mv=_mv, dtype=np.intp):
    size = nextx.size
    nrow, ncol = nextx.shape[0], nextx.shape[-1]
    nextx_flat = nextx.ravel()
    nexty_flat = nexty.ravel()
    # allocate output arrays
    pits_lst = []
    idxs_ds = np.full(nextx.size, core._mv, dtype=dtype)
    n = 0
    for idx0 in range(nextx.size):
        if nextx_flat[idx0] == _mv:
            continue
        c1 = nextx_flat[idx0]
        r1 = nexty_flat[idx0]
        pit = ispit(c1) or ispit(r1)
        # convert from one- to zero-based index
        r_ds, c_ds = np.intp(r1 - 1), np.intp(c1 - 1)
        outside = r_ds >= nrow or c_ds >= ncol or r_ds < 0 or c_ds < 0
        idx_ds = c_ds + r_ds * ncol
        # pit or outside or ds cell is mv
        if pit or outside or nextx_flat[idx_ds] == _mv:
            pits_lst.append(idx0)
            idxs_ds[idx0] = idx0
        else:
            idxs_ds[idx0] = idx_ds
        n += 1
    return idxs_ds, np.array(pits_lst, dtype=dtype), n


@njit
def _to_array(idxs_ds, shape, mv=core._mv):
    """convert 1D index to 3D NEXTXY raster"""
    ncol = shape[1]
    nextx = np.full(idxs_ds.size, _mv, dtype=np.int32)
    nexty = np.full(idxs_ds.size, _mv, dtype=np.int32)
    for idx0 in range(idxs_ds.size):
        idx_ds = idxs_ds[idx0]
        if idx_ds == mv:
            continue
        elif idx0 == idx_ds:  # pit
            nextx[idx0] = _pv[0]
            nexty[idx0] = _pv[0]
        else:
            # convert idx_ds to one-based row / col indices
            nextx[idx0] = idx_ds % ncol + 1
            nexty[idx0] = idx_ds // ncol + 1
    return nextx.reshape(shape), nexty.reshape(shape)


def isvalid(flwdir):
    """True if NEXTXY raster is valid"""
    isfmt1 = isinstance(flwdir, tuple) and len(flwdir) == 2
    isfmt2 = (
        isinstance(flwdir, np.ndarray) and flwdir.ndim == 3 and flwdir.shape[0] == 2
    )
    if not (isfmt1 or isfmt2):
        return False
    nextx, nexty = flwdir  # should work for [2,:,:] and ([:,:], [:,:])
    mask = np.logical_or(isnodata(nextx), ispit(nextx))
    return (
        nexty.dtype == "int32"
        and nextx.dtype == "int32"
        and np.all(nexty.shape == nextx.shape)
        and np.all(nextx[~mask] >= 0)
        and np.all(nextx[mask] == nexty[mask])
    )


@njit
def ispit(dd, _pv=_pv):
    """True if NEXTXY pit"""
    return np.logical_or(dd == _pv[0], dd == _pv[1])


@njit
def isnodata(dd):
    """True if NEXTXY nodata"""
    return dd == _mv


def read_nextxy(fn: Union[str, Path], nrow: int, ncol: int, bbox: List) -> np.ndarray:
    """Read nextxy data from binary file.

    Parameters
    ----------
    fn : str, Path
        Path to nextxy.bin file
    nrow, ncol : int
        Number or rows and columns in nextxy file.
    bbox: list of float
        domain bounding box [xmin, ymin, xmax, ymax]

    Returns
    -------
    np.ndarray
        Nextxy data
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
    """
    data = np.fromfile(fn, "i4").reshape(2, nrow, ncol)
    assert len(bbox) == 4, "Bounding box should contain 4 coordinates."
    transform = gis_utils.transform_from_bounds(*bbox, ncol, nrow)
    return data, transform

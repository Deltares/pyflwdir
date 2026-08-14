"""Implementation of NEXTXY flow direction type and methods.

This type is mainly used for the CaMa-Flood model.
Note that X (column) and Y (row) coordinates are one-based.
"""

from pathlib import Path

import numpy as np
from affine import Affine
from numba import njit

from . import core, gis_utils

__all__ = ["read_nextxy"]

# NEXTXY type
_ftype = "nextxy"
_mv: int = np.int32(-9999)  # type: ignore[assignment]
# -10 is inland termination, -9 river outlet at ocean
_pv = np.array([-9, -10], dtype=np.int32)
# NOTE: data below for consistency with LDD / D8 types and testing
_us = np.ones((2, 3, 3), dtype=np.int32) * 2
_us[:, 1, 1] = _pv[0]


def from_array(
    flwdir: np.ndarray | tuple, dtype: type = np.intp
) -> tuple[np.ndarray, np.ndarray, int]:
    if not (
        (isinstance(flwdir, tuple) and len(flwdir) == 2)
        or (
            isinstance(flwdir, np.ndarray) and flwdir.ndim == 3 and flwdir.shape[0] == 2
        )
    ):
        raise TypeError("NEXTXY flwdir data not understood")
    nextx, nexty = flwdir  # convert [2,:,:] OR ([:,:], [:,:]) to [:,:], [:,:]
    return _from_array(nextx, nexty, dtype=dtype)


def to_array(
    idxs_ds: np.ndarray, shape: tuple[int, int], mv: int = core._mv
) -> np.ndarray:
    nextx, nexty = _to_array(idxs_ds, shape, mv=mv)
    return np.stack([nextx, nexty])


@njit(cache=True)
def _from_array(
    nextx: np.ndarray,
    nexty: np.ndarray,
    _mv: int = _mv,
    dtype: type = np.intp,
) -> tuple[np.ndarray, np.ndarray, int]:
    size = nextx.size
    nrow, ncol = nextx.shape[0], nextx.shape[-1]
    nextx_flat = nextx.ravel()
    nexty_flat = nexty.ravel()
    # allocate output arrays
    pits_lst: list = []
    idxs_ds: np.ndarray = np.full(nextx.size, core._mv, dtype=dtype)
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


@njit(cache=True)
def _to_array(
    idxs_ds: np.ndarray, shape: tuple[int, int], mv: int = core._mv
) -> tuple[np.ndarray, np.ndarray]:
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


def isvalid(flwdir: np.ndarray | tuple) -> bool:
    """True if NEXTXY raster is valid"""
    isfmt1 = isinstance(flwdir, tuple) and len(flwdir) == 2
    isfmt2 = (
        isinstance(flwdir, np.ndarray) and flwdir.ndim == 3 and flwdir.shape[0] == 2
    )
    if not (isfmt1 or isfmt2):
        return False
    nextx, nexty = flwdir  # should work for [2,:,:] and ([:,:], [:,:])
    mask = np.logical_or(isnodata(nextx), ispit(nextx))
    return bool(
        nexty.dtype == "int32"
        and nextx.dtype == "int32"
        and np.all(nexty.shape == nextx.shape)
        and np.all(nextx[~mask] >= 0)
        and np.all(nextx[mask] == nexty[mask])
    )


@njit(cache=True)
def ispit(dd: np.ndarray | int, _pv: np.ndarray = _pv) -> np.ndarray | bool:
    """True if NEXTXY pit"""
    return np.logical_or(dd == _pv[0], dd == _pv[1])


@njit(cache=True)
def isnodata(dd: np.ndarray | int) -> np.ndarray | bool:
    """True if NEXTXY nodata"""
    return dd == _mv


def read_nextxy(
    fn: str | Path, nrow: int, ncol: int, bbox: list
) -> tuple[np.ndarray, Affine]:
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
    west, south, east, north = bbox
    transform = gis_utils.transform_from_bounds(west, south, east, north, ncol, nrow)
    return data, transform

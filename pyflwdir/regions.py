# -- coding: utf-8 --
"""Methods to for regions, i.e. connected areas with same unique ID.
Building on scipy.ndimage measurement methods, see
https://docs.scipy.org/doc/scipy/reference/ndimage.html#measurements
"""

from scipy import ndimage
import numpy as np
from numba import njit

from pyflwdir import gis_utils

__all__ = ["region_bounds", "region_slices", "region_sum", "region_area"]


def region_sum(data, regions):
    """Returns the sum of values in `data` for each unique label in `regions`.

    NOTE: a region must be a connected area with the same ID,
    where ID are integer values larger than zero.

    Parameters
    ----------
    data: 2D array
        input data
    regions: 2D array of int
        raster with unique IDs for each region, must have the same shape as `data`.

    Returns
    -------
    lbs, sum: 1D array
        arrays of the unique region IDs, and associated sum of input data
    """
    lbs = np.unique(regions[regions > 0])
    return lbs, ndimage.sum(data, regions, index=lbs)


def region_area(regions, transform=gis_utils.IDENTITY, latlon=False):
    """Returns the area [m2] for each unique label in `regions`.

    NOTE: a region must be a connected area with the same ID,
    where ID are integer values larger than zero.

    Parameters
    ----------
    regions: 2D array of int
        raster with unique IDs for each region, must have the same shape as `data`.
    latlon: bool
        True for geographic CRS, False for projected CRS.
        If True, the transform units are assumed to be degrees and converted to metric distances.
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.

    Returns
    -------
    lbs, areas: 1D array
        array of the unique region IDs, and associated areas [m2]
    """
    area = gis_utils.area_grid(transform=transform, shape=regions.shape, latlon=latlon)
    return region_sum(area, regions)


def region_slices(regions):
    """Returns slices for each unique label in `regions`.

    NOTE: a region must be a connected area with the same ID,
    where ID are integer values larger than zero.

    Parameters
    ----------
    regions: 2D array of int
        raster with unique IDs for each region, must have the same shape as `data`.

    Returns
    -------
    lbs: 1D array
        array of the unique region IDs
    slices: list of tuples
        Each tuple contains slices, one for each dimension
    """
    lbs = np.unique(regions[regions > 0])
    if lbs.size == 0:
        raise ValueError("No regions found in data")
    slices = ndimage.find_objects(regions)
    slices = [s for s in slices if s is not None]
    return lbs, slices


def region_bounds(regions, transform=gis_utils.IDENTITY):
    """Returns the bounding box each unique label in `regions`.

    NOTE: a region must be a connected area with the same ID,
    where ID are integer values larger than zero.

    Parameters
    ----------
    regions: 2D array of int
        raster with unique IDs for each region, must have the same shape as `data`.
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.

    Returns
    -------
    lbs: 1D array
        array of the unique region IDs
    bboxs: 2D array with shape (lbs.size, 4)
        bounding box [xmin, ymin, xmax, ymax] for each label
    total_bbox: 1D array
        total bounding box of all regions
    """
    lbs, slices = region_slices(regions)
    xres, yres = transform[0], transform[4]
    lons, lats = gis_utils.affine_to_coords(transform, regions.shape)
    iy = np.array([0, -1])
    ix = iy.copy()
    if yres < 0:
        iy = iy[::-1]
    if xres < 0:
        ix = ix[::-1]
    dx = np.abs(xres) / 2
    dy = np.abs(yres) / 2
    bboxs = []
    for yslice, xslice in slices:
        xmin, xmax = lons[xslice][ix]
        ymin, ymax = lats[yslice][iy]
        bboxs.append([xmin - dx, ymin - dy, xmax + dx, ymax + dy])
    bboxs = np.asarray(bboxs)
    total_bbox = np.hstack([bboxs[:, :2].min(axis=0), bboxs[:, 2:].max(axis=0)])
    return lbs, bboxs, total_bbox


@njit
def region_outlets(regions, idxs_ds, seq):
    """Returns the linear index of the outlet cell in `regions`.

    NOTE: a region must be a connected area with the same ID,
    where ID are integer values larger than zero.

    Parameters
    ----------
    regions: 2D array of int
        raster with unique IDs for each region, must have the same shape as `data`.
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream

    Returns
    -------
    lbs: 1D array
        array of the unique region IDs
    idxs_out: 1D array
        linear index of outlet cell per region
    """
    lbs_lst, idxs_lst = [], []
    for idx in seq[::-1]:  # up- to downstream
        idx_ds = idxs_ds[idx]
        lb0 = regions[idx]
        # outlet: idx inside region (lb0) and idx_ds outside region or pit
        if lb0 > 0 and (idx_ds == idx or regions[idx_ds] != lb0):
            idxs_lst.append(idx)
            lbs_lst.append(lb0)
    lbs = np.array(lbs_lst, dtype=regions.dtype)
    idxs_out = np.array(idxs_lst, dtype=idxs_ds.dtype)
    sort = np.argsort(lbs)
    return lbs[sort], idxs_out[sort]

# -- coding: utf-8 --
"""Methods to for regions, i.e. connected areas with same unique ID.
Building on scipy.ndimage measurement methods, see
https://docs.scipy.org/doc/scipy/reference/ndimage.html#measurements
"""

from scipy import ndimage
import numpy as np
from numba import njit

from . import gis_utils

__all__ = ["region_bounds", "region_slices", "region_sum", "region_area"]


def region_sum(data, regions):
    """Returns the sum of values in `data` for each unique label in `regions`.

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
    if regions.ndim != 2:
        raise ValueError('The "regions" array should be two dimensional')
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
    regions_flat = regions.ravel()
    lbs_lst, idxs_lst = [], []
    for idx in seq[::-1]:  # up- to downstream
        idx_ds = idxs_ds[idx]
        lb0 = regions_flat[idx]
        # outlet: idx inside region (lb0) and idx_ds outside region or pit
        if lb0 > 0 and (idx_ds == idx or regions_flat[idx_ds] != lb0):
            idxs_lst.append(idx)
            lbs_lst.append(lb0)
    lbs = np.array(lbs_lst, dtype=regions.dtype)
    idxs_out = np.array(idxs_lst, dtype=idxs_ds.dtype)
    sort = np.argsort(lbs)
    return lbs[sort], idxs_out[sort]


def region_dissolve(
    regions,
    labels=None,
    idxs=None,
    transform=gis_utils.IDENTITY,
    latlon=False,
    **kwargs,
):
    """Dissolve regions into its nearest neighboring regions.

    Regions to be dissolved are provided by either their `labels` or one location
    per region expressed with a linear index in `idxs`. These regions are assigned the
    label of the nearest neighboring region. If a locations `idxs` are provided the
    proximitity to other regions from that location. This can be  usefull to e.g.
    dissolve basins based on the distance from its outlet.

    Parameters
    ----------
    regions: 2D-array of int
        raster with unique non-zero positive IDs for each region
    labels: 1D-array of int
        labels of regions to be dissolved. Must be unique and larger than zero.
    idxs: 1D-array of int
        linear index of one location per region to be dissolved
    latlon: bool
        True for geographic CRS, False for projected CRS.
        If True, the transform units are assumed to be degrees and converted to metric distances.
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.


    Returns
    -------
    basins_out : 2D-array of int
        raster with basin IDs
    """
    if idxs is not None and labels is None:
        labels = regions.flat[idxs]
    elif labels is not None and idxs is None:
        labels = np.atleast_1d(labels)
    else:
        raise ValueError('Either "labels" or "idxs" must be provided.')
    if np.unique(labels[labels > 0]).size != labels.size:
        raise ValueError("Found non-unique or zero-value labels.")
    if regions.ndim != 2:
        raise ValueError('The "regions" array should be two dimensional')
    # set regions to be dissolved to zero (=background value)
    # and spread labels of valid regions
    regions0 = regions.copy()
    regions0[np.isin(regions, labels)] = 0
    assert np.any(regions0 != 0)
    out, _, dst = gis_utils.spread2d(
        regions0, nodata=0, transform=transform, latlon=latlon, **kwargs
    )
    if idxs is None:  # get idxs based on smallest distance per region
        r, c = zip(*ndimage.minimum_position(dst, regions, labels))
        idxs = np.asarray(r) * regions.shape[1] + np.asarray(c)
    # read labels of nearest regions at idxs
    labels1 = out.flat[idxs]
    # relabel regions
    d = {old: new for old, new in zip(labels, labels1)}
    return np.vectorize(lambda x: d.get(x, x))(regions)

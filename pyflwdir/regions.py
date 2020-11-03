# -- coding: utf-8 --
""""""

from scipy import ndimage
import numpy as np
from numba import njit

from pyflwdir import gis_utils

__all__ = ["region_bounds", "region_slices"]


def region_sum(data, regions):
    lbs = np.unique(regions[regions > 0])
    return lbs, ndimage.sum(data, regions, index=lbs)


def region_area(regions, transform=gis_utils.IDENTITY, latlon=False):
    if latlon:
        lon, lat = gis_utils.affine_to_coords(transform, regions.shape)
        area = gis_utils.reggrid_area(lat, lon)
    else:
        area = np.ones(regions.shape, dtype=np.float32) * abs(
            transform[0] * transform[4]
        )
    return region_sum(area, regions)


def region_slices(regions):
    lbs = np.unique(regions[regions > 0])
    if lbs.size == 0:
        raise ValueError("No regions found in data")
    slices = ndimage.find_objects(regions)
    slices = [s for s in slices if s is not None]
    return lbs, slices


def region_bounds(regions, transform=gis_utils.IDENTITY):
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

# -- coding: utf-8 --
from scipy import ndimage
import pandas as pd
import numpy as np
from numba import njit

from pyflwdir import gis_utils

# TODO check if basin in valid labeled array


def basin_sum(data, basins):
    lbs = np.unique(basins[basins > 0])
    return ndimage.sum(data, basins, index=lbs)


def basin_area(basins, affine=gis_utils.IDENTITY, latlon=False):
    if latlon:
        lon, lat = gis_utils.affine_to_coords(affine, basins.shape)
        area = gis_utils.reggrid_area(lat, lon)
    else:
        area = np.ones(basins.shape, dtype=np.float32) * abs(affine[0] * affine[4])
    return basin_sum(area, basins)


def basin_slices(basins):
    lbs = np.unique(basins[basins > 0])
    slices = ndimage.find_objects(basins)
    df = pd.DataFrame(
        index=lbs,
        data=[s for s in slices if s is not None],
        columns=["yslice", "xslice"],
    )
    return df


def basin_bounds(basins, affine=gis_utils.IDENTITY):
    df = basin_slices(basins)
    xres, yres = affine[0], affine[4]
    lons, lats = gis_utils.affine_to_coords(affine, basins.shape)
    xs = np.array([(s.start, s.stop) for s in df["xslice"]])
    ys = np.array([(s.start, s.stop) for s in df["yslice"]])
    if yres < 0:
        df["ymax"], df["ymin"] = lats[ys[:, 0]], lats[ys[:, 1] - 1]
    else:
        df["ymin"], df["ymax"] = lats[ys[:, 0]], lats[ys[:, 1] - 1]
    if xres < 0:
        df["xmax"], df["xmin"] = lons[xs[:, 0]], lons[xs[:, 1] - 1]
    else:
        df["xmin"], df["xmax"] = lons[xs[:, 0]], lons[xs[:, 1] - 1]
    df["xmin"] -= abs(xres) / 2
    df["ymin"] -= abs(yres) / 2
    df["xmax"] += abs(xres) / 2
    df["ymax"] += abs(yres) / 2
    return df.drop(columns=["yslice", "xslice"])


def total_basin_bounds(basins, affine=gis_utils.IDENTITY):
    b = basin_bounds(basins, affine=affine)
    bbox = np.array(
        [b["xmin"].min(), b["ymin"].min(), b["xmax"].max(), b["ymax"].max(),]
    )
    return bbox

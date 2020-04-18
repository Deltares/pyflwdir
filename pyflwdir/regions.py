# -- coding: utf-8 --
""""""

from scipy import ndimage
import pandas as pd
import numpy as np
from numba import njit

from pyflwdir import gis_utils

__all__ = ["region_bounds", "region_slices"]


def region_sum(data, basins):
    lbs = np.unique(basins[basins > 0])
    return ndimage.sum(data, basins, index=lbs)


def region_area(basins, transform=gis_utils.IDENTITY, latlon=False):
    if latlon:
        lon, lat = gis_utils.affine_to_coords(transform, basins.shape)
        area = gis_utils.reggrid_area(lat, lon)
    else:
        area = np.ones(basins.shape, dtype=np.float32) * abs(
            transform[0] * transform[4]
        )
    return region_sum(area, basins)


def region_slices(basins):
    lbs = np.unique(basins[basins > 0])
    slices = ndimage.find_objects(basins)
    df = pd.DataFrame(
        index=lbs,
        data=[s for s in slices if s is not None],
        columns=["yslice", "xslice"],
    )
    return df


def region_bounds(basins, transform=gis_utils.IDENTITY):
    df = region_slices(basins)
    xres, yres = transform[0], transform[4]
    lons, lats = gis_utils.affine_to_coords(transform, basins.shape)
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
    df = df.drop(columns=["yslice", "xslice"])
    df.loc[0, ["xmin", "ymin", "xmax", "ymax"]] = [
        df["xmin"].min(),
        df["ymin"].min(),
        df["xmax"].max(),
        df["ymax"].max(),
    ]
    return df[["xmin", "ymin", "xmax", "ymax"]].sort_index()


def total_region_bounds(basins, transform=gis_utils.IDENTITY):
    b = region_bounds(basins, transform=transform)
    bbox = np.array(
        [b["xmin"].min(), b["ymin"].min(), b["xmax"].max(), b["ymax"].max(),]
    )
    return bbox

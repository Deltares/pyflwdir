# -- coding: utf-8 --
""""""

from scipy import ndimage
import pandas as pd
import numpy as np
from numba import njit

from pyflwdir import gis_utils

__all__ = ["region_bounds", "region_slices"]


def region_sum(data, regions):
    lbs = np.unique(regions[regions > 0])
    return ndimage.sum(data, regions, index=lbs)


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
    df = pd.DataFrame(
        index=lbs,
        data=[s for s in slices if s is not None],
        columns=["yslice", "xslice"],
    )
    return df


def region_bounds(regions, transform=gis_utils.IDENTITY):
    df = region_slices(regions)
    xres, yres = transform[0], transform[4]
    lons, lats = gis_utils.affine_to_coords(transform, regions.shape)
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


def total_region_bounds(regions, transform=gis_utils.IDENTITY):
    b = region_bounds(regions, transform=transform)
    bbox = np.array(
        [
            b["xmin"].min(),
            b["ymin"].min(),
            b["xmax"].max(),
            b["ymax"].max(),
        ]
    )
    return bbox

# -- coding: utf-8 --
from scipy import ndimage
import pandas as pd
import numpy as np
from numba import njit

from pyflwdir import gis_utils

def basin_sum(data, basins):
    lbs = np.unique(basins[basins>0])
    df_sum = pd.DataFrame(
        index = lbs,
        data = ndimage.sum(data, basins, index=lbs),
        columns = ['sum']
    )
    return df_sum

def basin_area(basins, affine):
    height, width = basins.shape
    lon, lat = gis_utils.affine_to_coords(affine, width, height)
    area = gis_utils.reggrid_area(lat, lon)
    df_area = basin_sum(area, basins)
    df_area.columns = ['area']
    return df_area

def basin_slices(basins):
    lbs = np.unique(basins[basins>0])
    slices = ndimage.find_objects(basins)
    slices = [s for s in slices if s is not None]
    df_slices = pd.DataFrame(
        index = lbs,
        data = slices,
        columns = ['yslice', 'xslice']
    )
    return df_slices

def basin_bounds(basins, affine=gis_utils.IDENTITY):
    df_slices = basin_slices(basins)
    height, width = basins.shape
    lons, lats = gis_utils.affine_to_coords(affine, width, height)
    xres, yres = affine[0], affine[4]
    if yres < 0:
        lats = lats[::-1]
        yres = -yres
    bboxs = np.zeros((df_slices.index.size, 4), dtype=np.float64)
    for i in range(bboxs.shape[0]):
        yslice, xslice = df_slices.iloc[i,:] 
        xmin, xmax = lons[xslice][[0,-1]]
        ymin, ymax = lats[yslice][[0,-1]]
        bboxs[i,:] = xmin-xres/2., ymin-yres/2., xmax+xres/2., ymax+yres/2.
    df_bounds = pd.DataFrame(
        index = df_slices.index,
        data = bboxs,
        columns = ['xmin', 'ymin', 'xmax', 'ymax']
    )
    return df_bounds

def total_basin_bounds(basins, affine=gis_utils.IDENTITY):
    b = basin_bounds(basins, affine=affine)
    bbox = np.array([
            b['xmin'].min(),  
            b['ymin'].min(),  
            b['xmax'].max(),  
            b['ymax'].max(), 
    ])
    return bbox
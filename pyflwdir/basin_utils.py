# -- coding: utf-8 --
from scipy import ndimage
import pandas as pd
import numpy as np
from numba import njit

from pyflwdir import gis_utils

#TODO check if basin in valid labeled array

def basin_sum(data, basins):
    lbs = np.unique(basins[basins>0])
    return ndimage.sum(data, basins, index=lbs)

def basin_area(basins, affine):
    lon, lat = gis_utils.affine_to_coords(affine, basins.shape)
    area = gis_utils.reggrid_area(lat, lon)
    return basin_sum(area, basins)

def basin_slices(basins):
    lbs = np.unique(basins[basins>0])
    slices = ndimage.find_objects(basins)
    df_slices = pd.DataFrame(
        index = lbs,
        data = [s for s in slices if s is not None],
        columns = ['yslice', 'xslice']
    )
    return df_slices

def basin_bounds(basins, affine=gis_utils.IDENTITY):
    df_slices = basin_slices(basins)
    lons, lats = gis_utils.affine_to_coords(affine, basins.shape)
    xres, yres = affine[0], affine[4]
    bboxs = np.zeros((df_slices.index.size, 4), dtype=np.float64)
    for i, idx in enumerate(df_slices.index):
        yslice, xslice = df_slices.loc[idx,['yslice', 'xslice']] 
        if xres < 0:
            xmax, xmin = lons[xslice][[0,-1]]
        else:
            xmin, xmax = lons[xslice][[0,-1]]
        if yres < 0:
            ymax, ymin = lats[yslice][[0,-1]]
        else:
            ymin, ymax = lats[yslice][[0,-1]]
    bboxs[i,:] = xmin-abs(xres)/2., ymin-abs(yres)/2., xmax+abs(xres)/2., ymax+abs(yres)/2.
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
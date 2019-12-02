# -*- coding: utf-8 -*-
# Author: Willem van Verseveld @Deltares
# August 2019

from numba import njit
import numpy as np
import math
from pyflwdir import gis_utils

@njit
def slope(dem, nodata, latlon=False, affine=gis_utils.IDENTITY):
    
    xres, yres, north = affine[0], affine[4], affine[5]
    slope = np.zeros(dem.shape,dtype=np.float32)    
    nrow, ncol = dem.shape
    
    elev = np.zeros((3,3), dtype=dem.dtype)

    for r in range(0,nrow):
        for c in range(0,ncol):
            if dem[r,c] != nodata:
                # start with matrix based on central value (inside loop)
                elev[:,:] = dem[r,c]
            
                for dr in range(-1, 2):
                    row = r + dr
                    i = dr + 1
                    if row >= 0 and row < nrow:
                        for dc in range(-1, 2):
                            col = c + dc
                            j = dc + 1
                            if col >= 0 and col < ncol:
                                # fill matrix with elevation, except when nodata
                                if dem[row,col] != nodata:
                                    elev[i,j] = dem[row,col]

                dzdx = ((elev[0,0]+2*elev[1,0]+elev[2,0]) - (elev[0,2]+2*elev[1,2]+elev[2,2]))/ (8 * xres) 
                dzdy = ((elev[0,0]+2*elev[0,1]+elev[0,2]) - (elev[2,0]+2*elev[2,1]+elev[2,2]))/ (8 * np.abs(yres))

                if latlon:
                    lat = north + (r+0.5)*yres
                    deg_y = gis_utils.degree_metres_y(lat)
                    deg_x = gis_utils.degree_metres_x(lat)
                    slp = math.hypot(dzdx/deg_x, dzdy/deg_y)
                else:
                    slp = math.hypot(dzdx, dzdy)
            else:
                slp = nodata
            
            slope[r,c] = slp
        
    return slope
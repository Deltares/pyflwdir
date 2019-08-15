# -*- coding: utf-8 -*-
# Author: Willem van Verseveld @Deltares
# August 2019

from numba import njit
import numpy as np
import math

# import flow direction definition
from .gridtools import lat_to_dx, lat_to_dy

#NOTE: this assumes python Affine transfrom. should include check
@njit
def calc_slope(dem, nodata, sizeinmetres, transform):
    
    slope = np.zeros(dem.shape,dtype=np.float32)    
    nrow, ncol = dem.shape
    cellsize = transform[0]
    
    elev = np.zeros((3,3), dtype=dem.dtype)

    for r in range(0,nrow):
        for c in range(0,ncol):
            
            
            if dem[r,c] != nodata:
                # EDIT: start with matrix based on central value (inside loop)
                elev[:,:] = dem[r,c]
            
                for dr in range(-1, 2):
                    row = r + dr
                    i = dr + 1
                    if row >= 0 and row < nrow:
                        for dc in range(-1, 2):
                            col = c + dc
                            j = dc + 1
                            if col >= 0 and col < ncol:
                                # EDIT: fill matrix with elevation, except when nodata
                                if dem[row,col] != nodata:
                                    elev[i,j] = dem[row,col]

                dzdx = ((elev[0,0]+2*elev[1,0]+elev[2,0]) - (elev[0,2]+2*elev[1,2]+elev[2,2]))/ (8 * cellsize) 
                dzdy = ((elev[0,0]+2*elev[0,1]+elev[0,2]) - (elev[2,0]+2*elev[2,1]+elev[2,2]))/ (8 * cellsize)

                if sizeinmetres:
                    slp = math.hypot(dzdx, dzdy)
                else:
                    # EDIT: convert lat/lon to dy/dx in meters to calculate hypot
                    lat = transform[5] - 0.5*cellsize - r*cellsize
                    dy, dx = lat_to_dy(lat), lat_to_dx(lat)
                    slp = math.hypot(dzdx*dx, dzdy*dy)
            else:
                slp = nodata
            
            slope[r,c] = slp
        
    return slope
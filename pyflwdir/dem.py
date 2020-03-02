# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

import numpy as np
from numba import njit

from pyflwdir.core import downstream_all, _mv


@njit
def adjust_elevation(idxs_ds, idxs_us, tree, elevtn_sparse):
    """Given a flow direction map, remove pits in the elevation map.
    Algorithm based on Yamazaki et al. (2012)
    
    Yamazaki, D., Baugh, C. A., Bates, P. D., Kanae, S., Alsdorf, D. E. and 
    Oki, T.: Adjustment of a spaceborne DEM for use in floodplain hydrodynamic 
    modeling, J. Hydrol., 436-437, 81-91, doi:10.1016/j.jhydrol.2012.02.045,
    2012.
    """
    for i in range(len(tree)):
        idxs = tree[-i - 1]  # from up- to downstream
        for idx0 in idxs:
            if idxs_us[idx0, 0] != _mv:
                # has upstream neighbors
                continue
            # @ most upstream point, i.e. no upstream neighbors
            # get downstream indices
            idxs = downstream_all(idx0, idxs_ds)
            # fix elevation
            elevtn_sparse[idxs] = _adjust_elevation(elevtn_sparse[idxs])
    return elevtn_sparse


@njit
def _adjust_elevation(elevtn):
    """fix elevation on single streamline based on minimum modification
    elevtn oderdered from up- to downstream
    """
    zmin = elevtn[0]
    zmax = elevtn[0]
    valid = True
    for i in range(elevtn.size):
        zi = elevtn[i]
        if valid:
            if zi <= zmin:  # sloping down. do nothing
                zmin = zi
            else:  # new pit: downstream z > upstream z
                valid = False
                zmax = zi
                imax = i
                imin = i - 1
        else:
            if zi <= zmin or i + 1 == elevtn.size:  # end of pit area: FIX
                # option 1: dig -> zmod = zmin, for all values after pit
                idxs = np.arange(imin, i)  # TODO: check if i+1 if at end of streamline
                zmod = np.ones(idxs.size, dtype=elevtn.dtype) * zmin
                cost = np.sum(elevtn[idxs] - zmod)
                if imax - imin > 1:  # all options are equal when imax = imin + 1
                    # option 2: fill -> zmod = zmax, for all values smaller than zmax, previous to zmax
                    idxs2 = np.where(elevtn[:imax] <= zmax)[0]
                    zmod2 = np.ones(idxs2.size, dtype=elevtn.dtype) * zmax
                    cost2 = np.sum(zmod2 - elevtn[idxs2])
                    if cost2 < cost:
                        cost, idxs, zmod = cost2, idxs2, zmod2
                    # option 3: dig and fill -> zmin < zmod < zmax
                    idxs3 = np.where(
                        np.logical_and(elevtn[:i] >= zmin, elevtn[:i] <= zmax)
                    )[0]
                    zorg = elevtn[idxs3]
                    for z3 in np.unique(zorg):
                        if z3 > zmin and z3 < zmax:
                            zmod3 = np.ones(idxs3.size, dtype=elevtn.dtype) * z3
                            i0 = 0
                            i1 = zorg.size
                            while zorg[i0] > z3:  # elevtn > z3 from start can remain
                                zmod3[i0] = zorg[i0]
                                i0 += i
                            while zorg[i1] < z3:  # elevtn < z3 from end can remain
                                zmod3[i1] = zorg[i1]
                                i1 -= 1
                            cost3 = np.sum(np.abs(zmod3 - elevtn[idxs3]))
                            if cost3 < cost:
                                cost, idxs, zmod = cost3, idxs3, zmod3
                # adjust elevtn
                elevtn[idxs] = zmod
                zmin = zi
                valid = True
            elif zi >= zmax:  # between zmin and zmax (slope up) # get last imax (!)
                zmax = zi
                imax = i
    return elevtn

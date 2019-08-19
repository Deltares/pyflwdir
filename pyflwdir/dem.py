# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

import numpy as np
from numba import njit

# import flow direction definition
from .core import fd
from .network import _nbs_us

@njit
def hydrologically_adjust_elevation(idxs_ds, flwdir_flat, elevtn_flat, shape):
    """Given a flow direction map, remove pits in the elevation map.
    Algorithm based on Yamazaki et al. (2012)
    
    Yamazaki, D., Baugh, C. A., Bates, P. D., Kanae, S., Alsdorf, D. E. and 
    Oki, T.: Adjustment of a spaceborne DEM for use in floodplain hydrodynamic 
    modeling, J. Hydrol., 436-437, 81-91, doi:10.1016/j.jhydrol.2012.02.045,
    2012.
    """
    # get stream length to outlet at most upstream cells
    idxs0, upds0 = _dist_to_outlet_at_divide(idxs_ds.astype(np.uint32), flwdir_flat, shape)
    # loop from longest to shortest streamline
    seq = np.argsort(upds0)[::-1]
    assert upds0[seq[0]] > upds0[seq[-1]]
    for i in seq:
        idxs = _streamline(idxs0[i], flwdir_flat, shape)            # get streamlines
        elevtn_flat[idxs] = _fix_pits_streamline(elevtn_flat[idxs]) # fix elevation
    return elevtn_flat.reshape(shape)

@njit
def _dist_to_outlet_at_divide(idxs_ds, flwdir_flat, shape):
    """returns indices and distance (no of cell) to outlet at watershed divide"""
    size = shape[0]*shape[1]
    # intialize map
    updist = np.zeros(size, dtype=np.int32)
    idx_lst = []
    upd_lst = []
    # loop through flwdir map
    while True:
        nbs_us, valid = _nbs_us(idxs_ds, flwdir_flat, shape) # NOTE nbs_us has dtype uint32
        if np.all(valid==np.int8(0)):
            break
        for i in range(idxs_ds.size):
            idx_ds = idxs_ds[i]
            d = updist[idx_ds]
            if valid[i] == np.int8(0): # most upstream
                idx_lst.append(idx_ds)
                upd_lst.append(d)
            else:
                idxs_us = nbs_us[i,]
                for idx_us in idxs_us:
                    if idx_us >= size: break
                    updist[idx_us] = d+1 # downstream length + 1
        # next iter
        idxs_ds = nbs_us.ravel()
        idxs_ds = idxs_ds[idxs_ds < size]
    return np.array(idx_lst), np.array(upd_lst)

@njit
def _streamline(idx_us, flwdir_flat, shape):
    """return the indices of all pixels downstream from idx_us"""
    nodes = [idx_us]
    while True:
        idx_ds = fd.ds_index(idx_us, flwdir_flat, shape)
        if idx_ds == -1 or idx_us == idx_ds:
            break
        else:
            nodes.append(idx_ds)
        idx_us = idx_ds
    return np.array(nodes)


@njit
def _fix_pits_streamline(elevtn):
    """fix elevation on single streamline based on minimum modification"""
    zmin = elevtn[0]
    zmax = elevtn[0]
    valid = True
    for i in range(elevtn.size):
        zi = elevtn[i]
        if valid:
            if zi <= zmin: # sloping down. do nothing
                zmin = zi
            else: # new pit: downstream z > upstream z
                valid = False
                zmax = zi 
                imax = i
                imin = i-1
        else:
            if zi <= zmin or i+1 == elevtn.size: # end of pit area: FIX
                # option 1: dig -> zmod = zmin, for all values after pit
                idxs = np.arange(imin, i) # TODO: check if i+1 if at end of streamline
                zmod = np.ones(idxs.size, dtype=elevtn.dtype)*zmin
                cost = np.sum(elevtn[idxs] - zmod)
                if imax-imin > 1: # all options are equal when imax = imin + 1
                    # option 2: fill -> zmod = zmax, for all values smaller than zmax, previous to zmax
                    idxs2 = np.where(elevtn[:imax] <= zmax)[0] 
                    zmod2 = np.ones(idxs2.size, dtype=elevtn.dtype)*zmax
                    cost2 = np.sum(zmod2 - elevtn[idxs2])
                    if cost2 < cost:
                        cost, idxs, zmod = cost2, idxs2, zmod2
                    # option 3: dig and fill -> zmin < zmod < zmax
                    idxs3 = np.where(np.logical_and(elevtn[:i] >= zmin, elevtn[:i] <= zmax))[0]
                    zorg = elevtn[idxs3]
                    for z3 in np.unique(zorg): 
                        if z3 > zmin and z3 < zmax: 
                            zmod3 = np.ones(idxs3.size, dtype=elevtn.dtype)*z3
                            i0 = 0
                            i1 = zorg.size
                            while zorg[i0] > z3: # elevtn > z3 from start can remain
                                zmod3[i0] = zorg[i0]
                                i0 += i
                            while zorg[i1] < z3: # elevtn < z3 from end can remain
                                zmod3[i1] = zorg[i1]
                                i1 -= 1
                            cost3 = np.sum(np.abs(zmod3 - elevtn[idxs3]))
                            if cost3 < cost:
                                cost, idxs, zmod = cost3, idxs3, zmod3
                # adjust elevtn
                elevtn[idxs] = zmod
                zmin = zi
                valid = True
            elif zi >= zmax: # between zmin and zmax (slope up) # get last imax (!)
                zmax = zi 
                imax = i
    return elevtn
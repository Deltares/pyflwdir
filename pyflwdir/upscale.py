# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np

# import flow direction definition
# from .utils import flwdir_check
from pyflwdir import core
_mv = core._mv

#### GENERIC CONVENIENCE FUNCTIONS #### 
# naming convention
# row,  col,    index,  width
# ----------------------------------------------
# r,    c,      idx,    ncol     -> lowres cells
# subr, subc,   subidx, subncol  -> subgrid/highres cells
# ri,   ci,     ii,     cellsize -> location within lowres cell 

@njit
def subidx_2_idx(subidx, subncol, cellsize, ncol):
    """Returns the lowres index <idx> of subgrid cell index <subidx>"""
    r = (subidx // subncol) // cellsize
    c = (subidx %  subncol) // cellsize
    return r * ncol + c

@njit
def ii_2_subidx(ii, idx, subncol, cellsize, ncol):
    """Returns the subgrid index <subidx> lowres cell index <indx> and index within that cell <ii>"""
    r = idx // ncol * cellsize + ii // cellsize 
    c = idx %  ncol * cellsize + ii %  cellsize
    return r * subncol + c

@njit
def cell_edge(subidx, subncol, cellsize):
    """Returns True if highress cell <subidx> is on edge of lowres cell"""
    ri = (subidx // subncol) % cellsize
    ci = (subidx %  subncol) % cellsize
    return ri == 0 or ci == 0 or ri+1 == cellsize or ci+1 == cellsize # desribes edge

#### EFFECTIVE AREA METHOD ####

@njit
def effective_area(subidx, subncol, cellsize):
    """Returns True if highress cell <subidx> is inside the effective area"""
    R = cellsize / 2.
    ri = (subidx // subncol) % cellsize
    ci = (subidx %  subncol) % cellsize
    return (ri**0.5 + ci**0.5 <= R**0.5) or (ri <= 0.5) or (ci <= 0.5) # describes effective area

@njit
def eam_repcell(subidxs_ds, subidxs_valid, subuparea, subshape, shape, cellsize):
    """Returns subgrid representative cell indices of lowres cells
    according to the effective area method. 
    
    Parameters
    ----------
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_valid : ndarray of int
        highres raster indices of vaild cells
    subuparea : ndarray of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    raster with subgrid outlet indices : ndarray with size shape[0]*shape[1]
    """
    subncol = subshape[1]
    nrow, ncol = shape
    # allocate output and internal array
    subidxs_rep = np.ones(nrow*ncol, dtype=subidxs_valid.dtype)*_mv
    uparea = np.zeros(nrow*ncol, dtype=subuparea.dtype)
    # loop over valid indices
    for i in range(subidxs_valid.size):
        subidx = subidxs_valid[i]
        # TODO check if we actually need to include pits here
        # NOTE including pits in the effective area is different from the original
        ispit = subidxs_ds[i] == i # NOTE internal index
        eff_area = effective_area(subidx, subncol, cellsize)
        # check upstream area if cell ispit or at effective area
        if ispit or eff_area:
            idx = subidx_2_idx(subidx, subncol, cellsize, ncol)
            upa = subuparea[subidx]
            upa0 = uparea[idx]
            # cell with largest upstream area is representative cell
            if upa > upa0:
                uparea[idx] = upa
                subidxs_rep[idx] = subidx
    return subidxs_rep

@njit
def eam_outlets(subidxs_rep, subidxs_ds, subidxs_valid, subshape, shape, cellsize):
    """Returns subgrid outlet cell indices of lowres cells which are located
    at the edge of the lowres cell downstream of the representative cell
    according to the double effective area method. 
    
    Parameters
    ----------
    subidxs_rep : ndarray of int
        highres indices of representative cells with size shape[0]*shape[1]
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_valid : ndarray of int
        highres raster indices of vaild cells
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    raster with subgrid outlet indices : ndarray    
    """
    subnrow, subncol = subshape
    nrow, ncol = shape
    # internal indices
    n = subidxs_valid.size
    idxs_internal = np.ones(subnrow*subncol, np.uint32)*core._mv
    idxs_internal[subidxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output 
    subidxs_out = np.ones(nrow*ncol, dtype=subidxs_ds.dtype)*_mv
    # loop over rep cell indices
    for i in range(subidxs_rep.size):
        subidx = subidxs_rep[i]
        if subidx == _mv: 
            continue
        while not cell_edge(subidx, subncol, cellsize):
            # next downstream subgrid cell index; complicated because of use internal indices 
            subidx1 = subidxs_valid[subidxs_ds[idxs_internal[subidx]]] 
            if subidx1 == subidx: # pit
                break
            subidx = subidx1 # next iter
        subidxs_out[i] = subidx
    return subidxs_out

@njit
def eam_nextidx(subidxs_out, subidxs_ds, subidxs_valid, subshape, shape, cellsize):
    """Returns next downstream lowres index by tracing a representative cell to the 
    next downstream effective area according to the effective area method. 
    
    Parameters
    ----------
    subidxs_out : ndarray of int
        highres indices of outlet cells with size shape[0]*shape[1]
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_valid : ndarray of int
        highres raster indices of vaild cells
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    raster with subgrid outlet indices : ndarray    
    """
    subnrow, subncol = subshape
    nrow, ncol = shape
    # internal indices
    n = subidxs_valid.size
    idxs_internal = np.ones(subnrow*subncol, np.uint32)*core._mv
    idxs_internal[subidxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output 
    idxs_ds = np.ones(nrow*ncol, dtype=subidxs_ds.dtype)*_mv
    # loop over rep cell indices
    for i in range(subidxs_out.size):
        subidx = subidxs_out[i]
        if subidx == _mv: 
            continue
        # move one subgrid cell downstream into next lowres cell
        subidx = subidxs_valid[subidxs_ds[idxs_internal[subidx]]] 
        while not effective_area(subidx, subncol, cellsize):
            # next downstream subgrid cell index; complicated because of use internal indices 
            subidx1 = subidxs_valid[subidxs_ds[idxs_internal[subidx]]] 
            if subidx1 == subidx: # pit
                break
            subidx = subidx1 # next iter
        idxs_ds[i] = subidx_2_idx(subidx, subncol, cellsize, ncol)
    return idxs_ds

def eam_upscale(subidxs_ds, subidxs_valid, subuparea, subshape, cellsize):
    """Returns the upscaled flow direction network based on the effective area method
    
    Parameters
    ----------
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_valid : ndarray of int
        highres raster indices of vaild cells
    subuparea : ndarray of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    raster with subgrid outlet indices : ndarray with size shape[0]*shape[1]
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow, ncol = int(np.ceil(subnrow/cellsize)), int(np.ceil(subncol/cellsize))
    shape = nrow, ncol
    # get representative cells    
    subidxs_rep = eam_repcell(subidxs_ds, subidxs_valid, subuparea, subshape, shape, cellsize)
    # get subgrid outlet cells
    subidxs_out = eam_outlets(subidxs_rep, subidxs_ds, subidxs_valid, subshape, shape, cellsize)
    # get next downstream lowres index
    nextidx = eam_nextidx(subidxs_out, subidxs_ds, subidxs_valid, subshape, shape, cellsize)
    return nextidx, subidxs_out, shape

#### DOUBLE EFFECTIVE AREA METHOD ####


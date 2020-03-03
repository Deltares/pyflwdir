# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Methods for upscaling high res flow direction data to lower model 
resolutions."""

from numba import njit
import numpy as np

from pyflwdir import core, core_nextidx

_mv = core._mv
__all__ = []

# naming convention
# row,  col,    index,  width
# ----------------------------------------------
# r,    c,      idx,    ncol     -> lowres cells
# subr, subc,   subidx, subncol  -> subgrid/highres cells
# ri,   ci,     ii,     cellsize -> location within lowres cell


#### GENERIC CONVENIENCE FUNCTIONS ####
@njit
def subidx_2_idx(subidx, subncol, cellsize, ncol):
    """Returns the lowres index <idx> of subgrid cell index <subidx>."""
    r = (subidx // subncol) // cellsize
    c = (subidx % subncol) // cellsize
    return r * ncol + c


@njit
def ii_2_subidx(ii, idx, subncol, cellsize, ncol):
    """Returns the subgrid index <subidx> based on the 
    lowres cell index <idx> and index within that cell <ii>."""
    r = idx // ncol * cellsize + ii // cellsize
    c = idx % ncol * cellsize + ii % cellsize
    return r * subncol + c


@njit
def in_d8(idx0, idx_ds, ncol):
    """Returns True if inside 3x3 (current and 8 neighboring) cells."""
    cond1 = abs(idx_ds - idx0) <= 1  # west, east
    cond2 = abs(idx_ds - idx0 - ncol) <= 1  # south
    cond3 = abs(idx_ds - idx0 + ncol) <= 1  # north
    return cond1 or cond2 or cond3


@njit
def update_idxs_ds_us(idx0, idx_ds, idx_ds0, idxs_ds, idxs_us, idxs_sparse):
    """Updates idxs_ds and idxs_us based on new idx_ds for idx0"""
    n = idxs_us.shape[1]
    # update idxs_ds
    id0 = idxs_sparse[idx0]  # internal index
    ids = idxs_sparse[idx_ds]  # internal index
    idxs_ds[id0] = ids
    # update idxs_us at new idx_ds if not pit -> add idx0
    if idx0 != idx_ds:
        if np.all(idxs_us[ids, :] != _mv) and n < 8:
            idxs_us1 = np.full((idxs_ds.size, 8), _mv, dtype=idxs_us.dtype)
            idxs_us1[:, :n] = idxs_us
            idxs_us = idxs_us1
            n = 8
        elif np.all(idxs_us[ids, :] != _mv):
            raise AssertionError
        for i in range(n):
            if idxs_us[ids, i] == _mv:
                idxs_us[ids, i] = id0
                break
    # update idxs_us at old idx_ds0 -> remove idx0
    ids0 = idxs_sparse[idx_ds0]  # internal index
    idxs_us0 = np.full(n, _mv, dtype=np.uint32)
    j = 0
    for i in range(n):
        if idxs_us[ids0, i] == _mv:
            break
        elif idxs_us[ids0, i] != id0:
            idxs_us0[j] = idxs_us[ids0, i]
            j += 1
    idxs_us[ids0, :] = idxs_us0
    return idxs_ds, idxs_us


#### DOUBLE MAXIMUM METHOD ####


@njit
def cell_edge(subidx, subncol, cellsize):
    """Returns True if highres cell <subidx> is on edge of lowres cell"""
    ri = (subidx // subncol) % cellsize
    ci = (subidx % subncol) % cellsize
    return ri == 0 or ci == 0 or ri + 1 == cellsize or ci + 1 == cellsize


@njit
def map_celledge(subidxs_ds, subidxs_dense, subshape, cellsize):
    """Returns a map with ones on subgrid cells of lowres cell edges"""
    surbnrow, subncol = subshape
    # allocate output and internal array
    edges = np.full(surbnrow * subncol, -1, dtype=np.int8)
    # loop over valid indices
    for subidx in subidxs_dense:
        if cell_edge(subidx, subncol, cellsize):
            edges[subidx] = np.int8(1)
        else:
            edges[subidx] = np.int8(0)
    return edges.reshape(subshape)


@njit
def dmm_exitcell(subidxs_ds, subidxs_dense, subuparea, subshape, shape, cellsize):
    """Returns exit subgrid cell indices of lowres cells according to the 
    double maximum method (DMM). 
    
    Parameters
    ----------
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
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
    1D array of int 
        subgrid indices of representative cells
    """
    subncol = subshape[1]
    nrow, ncol = shape
    # allocate output and internal array
    subidxs_rep = np.full(nrow * ncol, _mv, dtype=subidxs_ds.dtype)
    uparea = np.zeros(nrow * ncol, dtype=subuparea.dtype)
    # loop over valid indices
    for i in range(subidxs_dense.size):
        subidx = subidxs_dense[i]
        # NOTE including pits in the edge area is different from the original
        ispit = subidxs_ds[i] == i  # NOTE internal index
        edge = cell_edge(subidx, subncol, cellsize)
        # check upstream area if cell ispit or at effective area
        if ispit or edge:
            idx = subidx_2_idx(subidx, subncol, cellsize, ncol)
            upa = subuparea[subidx]
            upa0 = uparea[idx]
            # cell with largest upstream area is representative cell
            if upa > upa0:
                uparea[idx] = upa
                subidxs_rep[idx] = subidx
    return subidxs_rep


@njit
def dmm_nextidx(subidxs_rep, subidxs_ds, subidxs_dense, subshape, shape, cellsize):
    """Returns next downstream lowres index by tracing a representative cell 
    to where it leaves a buffered area around the lowres cell according to the 
    double maximum method (DMM). 
    
    Parameters
    ----------
    subidxs_rep : ndarray of int
        highres indices of representative cells
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
        highres raster indices of vaild cells
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    ndarray of int
        lowres indices of next downstream cell
    """
    subnrow, subncol = subshape
    subsize = subnrow * subncol
    nrow, ncol = shape
    R = cellsize / 2
    # internal indices
    n = subidxs_dense.size
    subidxs_sparse = np.full(subsize, _mv, np.uint32)
    subidxs_sparse[subidxs_dense] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output
    idxs_ds = np.full(nrow * ncol, _mv, dtype=subidxs_ds.dtype)
    # loop over rep cell indices
    for idx0 in range(subidxs_rep.size):
        subidx = subidxs_rep[idx0]
        idx = idx0
        if subidx == _mv:
            continue
        # subgrid coordinates at center of offset lowres cell
        dr = (subidx // subncol) % cellsize // R
        dc = (subidx % subncol) % cellsize // R
        subr0 = (idx0 // ncol + dr) * cellsize - 0.5
        subc0 = (idx0 % ncol + dc) * cellsize - 0.5
        while True:
            # next downstream subgrid cell index; NOTE use internal indices
            subidx1 = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
            idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
            if subidx1 == subidx:  # pit
                break
            elif idx1 != idx0:  # outside offset lowres cell
                subr = subidx // subncol
                subc = subidx % subncol
                if abs(subr - subr0) > R or abs(subc - subc0) > R:
                    break
            # next iter
            subidx = subidx1
            idx = idx1
        idxs_ds[idx0] = idx
    return idxs_ds


def dmm(subidxs_ds, subidxs_dense, subuparea, subshape, cellsize):
    """Returns the upscaled next downstream index based on the 
    double maximum method (DMM) [1].

    ...[1] Olivera F, Lear M S, Famiglietti J S and Asante K 2002 
    "Extracting low-resolution river networks from high-resolution digital 
    elevation models" Water Resour. Res. 38 13-1-13–8 
    Online: http://doi.wiley.com/10.1029/2001WR000726
    
    Parameters
    ----------
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
        highres raster indices of vaild cells
    subuparea : ndarray of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres indices of next downstream 
        ndarray of int
    subgrid indices of representative cells
        ndarray of int
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow = int(np.ceil(subnrow / cellsize))
    ncol = int(np.ceil(subncol / cellsize))
    shape = nrow, ncol
    # get representative cells
    subidxs_rep = dmm_exitcell(
        subidxs_ds, subidxs_dense, subuparea, subshape, shape, cellsize
    )
    # get next downstream lowres index
    nextidx = dmm_nextidx(
        subidxs_rep, subidxs_ds, subidxs_dense, subshape, shape, cellsize
    )
    return nextidx.reshape(shape), subidxs_rep.reshape(shape)


#### EFFECTIVE AREA METHOD ####


@njit
def effective_area(subidx, subncol, cellsize):
    """Returns True if highress cell <subidx> is inside the effective area."""
    R = cellsize / 2.0
    offset = R - 0.5  # lowres center at cellsize/2 - 0.5
    ri = abs((subidx // subncol) % cellsize - offset)
    ci = abs((subidx % subncol) % cellsize - offset)
    # describes effective area
    ea = (ri ** 0.5 + ci ** 0.5 <= R ** 0.5) or (ri <= 0.5) or (ci <= 0.5)
    return ea


@njit
def map_effare(subidxs_ds, subidxs_dense, subshape, cellsize):
    """Returns a map with ones on subgrid cells of lowres effective area."""
    surbnrow, subncol = subshape
    # allocate output and internal array
    effare = np.full(surbnrow * subncol, -1, dtype=np.int8)
    # loop over valid indices
    for subidx in subidxs_dense:
        if effective_area(subidx, subncol, cellsize):
            effare[subidx] = np.int8(1)
        else:
            effare[subidx] = np.int8(0)
    return effare.reshape(subshape)


@njit
def eam_repcell(subidxs_ds, subidxs_dense, subuparea, subshape, shape, cellsize):
    """Returns representative subgrid cell indices of lowres cells
    according to the effective area method. 
    
    Parameters
    ----------
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
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
    subgrid representative cell indices : ndarray with size shape[0]*shape[1]
    """
    subncol = subshape[1]
    nrow, ncol = shape
    # allocate output and internal array
    subidxs_rep = np.full(nrow * ncol, _mv, dtype=subidxs_ds.dtype)
    uparea = np.zeros(nrow * ncol, dtype=subuparea.dtype)
    # loop over internal indices
    for i in range(subidxs_dense.size):
        subidx = subidxs_dense[i]  # subgrid index
        # NOTE including pits is different from the original EAM
        ispit = subidxs_ds[i] == i  # NOTE internal index
        eff_area = effective_area(subidx, subncol, cellsize)
        idx = subidx_2_idx(subidx, subncol, cellsize, ncol)
        # check upstream area if cell ispit or at effective area
        if ispit or eff_area:
            upa0 = uparea[idx]
            upa = subuparea[subidx]
            # cell with largest upstream area is representative cell
            if upa > upa0:
                uparea[idx] = upa
                subidxs_rep[idx] = subidx
    return subidxs_rep


@njit
def eam_nextidx(subidxs_rep, subidxs_ds, subidxs_dense, subshape, shape, cellsize):
    """Returns next downstream lowres index by tracing a representative cell to 
    the next downstream effective area according to the effective area method. 
    
    Parameters
    ----------
    subidxs_rep : ndarray of int
        highres indices of representative subgird cells
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
        highres raster indices of vaild cells
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres indices of next downstream cell : ndarray    
    """
    subnrow, subncol = subshape
    nrow, ncol = shape
    # internal indices
    n = subidxs_dense.size
    subidxs_sparse = np.full(subnrow * subncol, _mv, np.uint32)
    subidxs_sparse[subidxs_dense] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output
    idxs_ds = np.full(nrow * ncol, _mv, dtype=subidxs_ds.dtype)
    # loop over rep cell indices
    for idx0 in range(subidxs_rep.size):
        subidx = subidxs_rep[idx0]
        if subidx == _mv:
            continue
        while True:
            # next downstream subgrid cell index; NOTE use internal indices
            subidx1 = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
            idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
            if subidx1 == subidx:  # at pit
                break
            elif idx1 != idx0 and effective_area(subidx1, subncol, cellsize):
                # in d8 effective area
                break
            # next iter
            subidx = subidx1
        idxs_ds[idx0] = idx1
    return idxs_ds


def eam(subidxs_ds, subidxs_dense, subuparea, subshape, cellsize):
    """Returns the upscaled next downstream index based on the 
    effective area method (EAM) [1].

    ...[1] Yamazaki D, Masutomi Y, Oki T and Kanae S 2008 
    "An Improved Upscaling Method to Construct a Global River Map" APHW
    
    Parameters
    ----------
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
        highres raster indices of vaild cells
    subuparea : ndarray of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres indices of next downstream
        ndarray of int
    subgrid indices of representative cells
        ndarray of int
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow = int(np.ceil(subnrow / cellsize))
    ncol = int(np.ceil(subncol / cellsize))
    shape = nrow, ncol
    # get representative cells
    subidxs_rep = eam_repcell(
        subidxs_ds, subidxs_dense, subuparea, subshape, shape, cellsize
    )
    # get next downstream lowres index
    nextidx = eam_nextidx(
        subidxs_rep, subidxs_ds, subidxs_dense, subshape, shape, cellsize
    )
    return nextidx.reshape(shape), subidxs_rep.reshape(shape)


#### CONNECTING OUTLETS SCALING METHOD ####
@njit
def com_outlets(
    subidxs_rep,
    subidxs_ds,
    subidxs_dense,
    subuparea,
    subshape,
    shape,
    cellsize,
    min_stream_len=0,
):
    """Returns subgrid outlet cell indices of lowres cells which are located
    at the edge of the lowres cell downstream of the representative cell
    according to the connecting outlets method (COM). 

    NOTE: If <min_stream_len> is larger than zero, the outlet does not have 
    to be the the lowres cells edge.
    
    Parameters
    ----------
    subidxs_rep : ndarray of int
        highres indices of representative cells with size shape[0]*shape[1]
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
        highres raster indices of vaild cells
    subuparea : ndarray of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells
    min_stream_len : int
        minimum length (pixels) between outlet and previous confluence
        a confluence is determined by at least a doubling of upstream area

    Returns
    -------
    raster with subgrid outlet indices
        ndarray of int
    """
    subnrow, subncol = subshape
    nrow, ncol = shape
    # internal indices
    n = subidxs_dense.size
    subidxs_sparse = np.full(subnrow * subncol, _mv, np.uint32)
    subidxs_sparse[subidxs_dense] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output
    subidxs_out = np.full(nrow * ncol, _mv, dtype=subidxs_ds.dtype)
    # loop over rep cell indices
    for idx0 in range(subidxs_rep.size):
        subidx = subidxs_rep[idx0]
        if subidx == _mv:
            continue
        if min_stream_len > 0:
            subupa = subuparea[subidx]
            subidx_prev_stream = _mv
            stream_len = 0
        while True:
            # next downstream subgrid cell index; NOTE use internal indices
            subidx1 = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
            if min_stream_len > 0:
                subupa1 = subuparea[subidx1]
                if subupa1 > 2 * subupa:  # confluence
                    subidx_prev_stream = subidx
                    stream_len = 0
            # at outlet if next subgrid cell is in next lowres cell
            outlet = idx0 != subidx_2_idx(subidx1, subncol, cellsize, ncol)
            pit = subidx1 == subidx
            if outlet or pit:
                if (
                    min_stream_len > 0
                    and stream_len <= min_stream_len
                    and subidx_prev_stream != _mv
                ):
                    subidx = subidx_prev_stream
                break
            # next iter
            subidx = subidx1
            if min_stream_len > 0:
                subupa = subupa1
                stream_len += 1
        subidxs_out[idx0] = subidx
    return subidxs_out


@njit
def com_nextidx(subidxs_out, subidxs_ds, subidxs_dense, subshape, shape, cellsize):
    """Returns next downstream lowres index according to connecting outlets 
    method (COM). Every outlet subgrid cell is traced to the next downstream 
    subgrid outlet cell. If this lays outside d8, we fallback to the next 
    downstream effective area.
    
    Parameters
    ----------
    subidxs_out : ndarray of int
        highres indices of outlet cells with size shape[0]*shape[1]
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
        highres raster indices of vaild cells
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres indices of next downstream, disconnected cells 
        Tuple of ndarray    
    """
    subnrow, subncol = subshape
    nrow, ncol = shape
    # internal indices
    n = subidxs_dense.size
    subidxs_sparse = np.full(subnrow * subncol, _mv, np.uint32)
    subidxs_sparse[subidxs_dense] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output
    nextidx = np.full(nrow * ncol, _mv, dtype=subidxs_ds.dtype)
    idxs_fix_lst = list()
    # loop over outlet cell indices
    for idx0 in range(subidxs_out.size):
        subidx = subidxs_out[idx0]
        if subidx == _mv:
            continue
        subidx_ds = _mv
        while True:
            # next downstream subgrid cell index; NOTE use internal indices
            subidx1 = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
            idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
            if subidxs_out[idx1] == subidx1 or subidx1 == subidx:
                # at outlet or pit
                if not in_d8(idx0, idx1, ncol):  # outside d8 neighbors
                    # flag index, use first pass ea
                    idxs_fix_lst.append(np.uint32(idx0))
                else:
                    subidx_ds = subidx1
                    # incorrect ds connection where pit
                    if subidxs_out[idx1] != subidx1:
                        idxs_fix_lst.append(np.uint32(idx0))  # flag index
                break
            if subidx_ds == _mv and effective_area(subidx1, subncol, cellsize):
                subidx_ds = subidx1  # first pass effective area
            # next iter
            subidx = subidx1
        # assert subidx_ds != _mv
        nextidx[idx0] = subidx_2_idx(subidx_ds, subncol, cellsize, ncol)
    return nextidx, np.array(idxs_fix_lst, dtype=np.uint32)


@njit
def next_outlet(
    subidx,
    subidxs_dense,
    subidxs_ds,
    subidxs_sparse,
    subidxs_out,
    subncol,
    cellsize,
    ncol,
):
    """Return lowres and subgrid indices of next outlet"""
    while True:
        # next downstream subgrid cell index; NOTE use internal indices
        subidx1 = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
        idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
        outlet = subidx1 == subidxs_out[idx1]
        pit = subidx1 == subidx
        if outlet or pit:
            break
        # next iter
        subidx = subidx1
    return subidx1, idx1, outlet


@njit
def com_nextidx_iter2(
    nextidx,
    subidxs_out,
    idxs_fix,
    subidxs_ds,
    subidxs_dense,
    subuparea,
    subshape,
    shape,
    cellsize,
):
    """Second iteration to fix cells which are not connected in subgrid.
    
    Parameters
    ----------
    nextidx : ndarray of int
        lowres indices of next downstream cell
    subidxs_out : ndarray of int
        subgrid indices of outlet cells with size shape[0]*shape[1]
    idxs_fix : ndarray of int
        lowres indices of cells which are disconnected in subgrid
    subidxs_ds : ndarray of int
        internal subgrid indices of downstream cells
        NOTE these are internal indices for valid cells only
    subidxs_dense : ndarray of int
        subgrid raster indices of vaild cells
    subuparea : ndarray of int
        highres flattened upstream area array
    subshape : tuple of int
        subgrid (highres) raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    raster with subgrid outlet indices : ndarray    
    """
    subnrow, subncol = subshape
    subsize = subnrow * subncol
    nrow, ncol = shape
    size = nrow * ncol
    # parse nextidx
    idxs_dense, idxs_ds, _ = core_nextidx.from_array(nextidx.reshape(shape))
    idxs_us = core._idxs_us(idxs_ds)
    # internal indices
    n = idxs_dense.size
    idxs_sparse = np.full(size, _mv, np.uint32)
    idxs_sparse[idxs_dense] = np.array([i for i in range(n)], dtype=np.uint32)
    subn = subidxs_dense.size
    subidxs_sparse = np.full(subsize, _mv, np.uint32)
    subidxs_sparse[subidxs_dense] = np.array([i for i in range(subn)], dtype=np.uint32)
    # loop over unconnected cells from up to downstream
    upa_check = subuparea[subidxs_out[idxs_fix]]
    seq = np.argsort(upa_check)
    for i0 in seq:  # @0A lowres fix index loop
        nextiter = False
        idx00 = idxs_fix[i0]
        # print(idx00)

        # STEP 1: get downstream path with subgrid outlet indices
        idxs_lst = list()
        subidxs_lst = list()
        connected = False
        # read outlet index and move into the next lowres cell to initialize
        subidx = subidxs_out[idx00]
        idx_ds0 = nextidx[idx00]  # original next downstream cell
        subidx = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
        idx0 = subidx_2_idx(subidx, subncol, cellsize, ncol)
        while True:  # @1A while noy connected to original downstream cell
            while True:  # @1B subgrid loop - while not at outlet
                subidx1 = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
                idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                pit = subidx1 == subidx
                if pit or idx0 != idx1:  # check pit or outlet
                    # connected if:
                    # - next downstream lowres cell not in path &
                    # - current subgrid outlet cell same as original next outlet
                    if pit:
                        connected = True
                    elif subidx == subidxs_out[idx_ds0]:
                        if idx_ds0 in idxs_lst:
                            pass
                        else:
                            connected = True
                    # check if valid cell (required at edges)
                    if np.any(idxs_dense == idx0):
                        subidxs_lst.append(subidx)  # append subgrid outlet index
                        idxs_lst.append(idx0)  # append lowres index
                    # if at original outlet cell of idx0 -> update idx_ds0
                    if subidx == subidxs_out[idx0]:
                        idx_ds0 = nextidx[idx0]
                    break  # @1B
                # next iter @1B
                subidx = subidx1
            if connected:  # with original ds subgrid cell
                break  # @1A
            # next iter @1A
            idx0 = idx1
            subidx = subidx1
        if connected and subidx == subidxs_out[nextidx[idx00]]:
            # connection at first outlet -> already fixed
            continue  # @0A
        elif not connected:
            continue  # @0A

        # STEP 2: find original upstream connections
        idxs_us_lst = list()
        idxs_ds0 = np.unique(np.array(idxs_lst, dtype=idxs_fix.dtype))
        # assert np.all(idxs_sparse[idxs_ds0] != _mv)
        for idx_ds in idxs_ds0:  # @2A lowres us connections loop
            for i in idxs_us[idxs_sparse[idx_ds], :]:
                if i == _mv:
                    break  # @2A
                idx0 = idxs_dense[i]
                # skip upstream nodes wich are on path of step 1
                if subidxs_out[idx0] in subidxs_lst or idx0 == idx00:
                    continue  # @2A
                # append lowres index of upstream connection
                idxs_us_lst.append(idx0)

        # STEP 3: connect original upstream connections to outlets on path
        noutlets = len(subidxs_lst)
        idxs_us_conn_lst = list()  # first possible connection of lateral
        idxs_us_conn_lst1 = list()  # last possible connection of lateral
        for i in range(len(idxs_us_lst)):  # @3A lowres us connections loop
            idx0 = idxs_us_lst[i]
            subidx = subidxs_out[idx0]
            connected = False
            j0, j1 = 0, 0  # start and end connection to path
            # move into next cell to initialize
            subidx = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
            idx = idx0
            ii = 0
            while True and ii <= 10:  # @3B find connecting outlet in subidxs_lst
                subidx1 = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
                idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                if subidx == subidx1 or idx != idx1:  # if pit OR outlet
                    if not connected:
                        ii += 1
                    for j in range(j0, noutlets):  # @3C check outlet loop
                        if subidxs_lst[j] == subidx:  # connection on path
                            if not connected:
                                j0, j1, connected = j, j, True
                            elif in_d8(idx0, idx, ncol):
                                j1 = j
                            break  # @3C
                    if (j1 + 1 == noutlets) or subidx == subidx1:
                        break  # @3C
                # next iter @3B
                subidx = subidx1
                idx = idx1
            if connected:
                idxs_us_conn_lst.append(j0)
                idxs_us_conn_lst1.append(j1)
            else:
                idxs_us_conn_lst.append(noutlets - 1)
                idxs_us_conn_lst1.append(noutlets - 1)
        # sort from up to downstream
        idxs_us_conn = np.array(idxs_us_conn_lst, dtype=nextidx.dtype)
        seq1 = np.argsort(idxs_us_conn)
        idxs_us0 = np.array(idxs_us_lst, dtype=nextidx.dtype)[seq1]
        subidxs_ds0 = subidxs_out[nextidx[idxs_us0]]
        idxs_us_conn1 = np.array(idxs_us_conn_lst1, dtype=nextidx.dtype)[seq1]
        idxs_us_conn = idxs_us_conn[seq1]

        # STEP 4: connect the dots
        bottleneck = list()
        nbottlenecks = -1
        while len(bottleneck) > nbottlenecks:
            nextiter = False
            nbottlenecks = len(bottleneck)
            idx_out_lst = list()
            subidx0_out_lst = list()
            idx_ds_lst = list()
            idx_ds0_lst = list()
            idx0_lst = list()
            idx0 = idx00
            j0, k0 = 0, 0
            for j in range(noutlets):  # @4A lowres connecting loop
                if nextiter:
                    continue
                idx1 = idxs_lst[j]
                subidx_out1 = subidxs_lst[j]
                # check if not connected to ds pit
                pit1 = (
                    subidx_out1
                    == subidxs_dense[subidxs_ds[subidxs_sparse[subidx_out1]]]
                )
                if pit1 and len(idx_out_lst) == 0:
                    idxs_pit = np.where(subidxs_out == subidx_out1)[0]
                    if idxs_pit.size == 1 and in_d8(idx0, idxs_pit[0], ncol):
                        # previous (smaller) branch already claimed pit
                        if nextidx[idx0] != idxs_pit[0]:
                            idx_ds0_lst.append(nextidx[idx0])
                            idx0_lst.append(idx0)
                            idx_ds_lst.append(idxs_pit[0])
                            nextidx[idx0] = idxs_pit[0]
                            # print('pit - connect', idx0, idxs_pit[0])
                        break  # @4A
                    elif idxs_pit.size == 0:
                        # set pit and move outlet to neighboring cell
                        # NOTE pit outlet outside lowres cell !
                        if nextidx[idx0] != idx0:
                            idx_ds0_lst.append(nextidx[idx0])
                            idx0_lst.append(idx0)
                            idx_ds_lst.append(idx0)
                            nextidx[idx0] = idx0
                            # print('pit', idx0, idx0)
                        idx_out_lst.append(idx0)
                        subidx0_out_lst.append(subidxs_out[idx0])
                        subidxs_out[idx0] = subidx_out1
                        # print('pit - outlet', idx0)
                        break  # @4A
                    else:
                        nextiter = True
                        # print('pit - no connection', idx0)
                # check if ds lowres cell already edited to avoid loops
                if idx1 in idx_out_lst or idx1 in bottleneck:
                    d8 = False
                else:
                    d8 = in_d8(idx0, idx1, ncol)
                # check lateral connections
                ks_bool = np.logical_and(
                    idxs_us_conn[k0:] >= j0, idxs_us_conn[k0:] <= j
                )
                ks = np.where(ks_bool)[0] + k0
                lats = ks.size > 0
                nextlats = np.all(idxs_us_conn1[ks] > j) if lats else False
                # check if possible to maka a d8 connection downstream, but
                # before next outlet
                nextd8 = False
                for jj in range(j + 1, noutlets):
                    idx = idxs_lst[jj]
                    if idx in idx_out_lst or idx in bottleneck:
                        continue
                    elif in_d8(idx0, idx, ncol):
                        nextd8 = True
                    if subidxs_out[idx] == subidxs_lst[jj]:  # original outlet
                        break
                # if next subgrid outlet exists nextd8 is False -> force update
                nextd8 = nextd8 and subidxs_out[idx1] != subidx_out1
                # print(idx0, idx1)
                # print(d8, nextd8, lats, nextlats)
                if not d8 and not nextd8:
                    nextiter = True
                elif (not lats and nextd8) or (nextlats and nextd8):
                    continue
                # UPDATE CONNECTIONS
                if (d8 and lats) or (d8 and not nextd8):
                    # update MAIN connection
                    if nextidx[idx0] != idx1:
                        idx_ds0_lst.append(nextidx[idx0])
                        idx0_lst.append(idx0)
                        idx_ds_lst.append(idx1)
                        nextidx[idx0] = idx1
                        # print('ds', idx0, idx1)
                    if subidx_out1 != subidxs_out[idx1]:
                        idx_out_lst.append(idx1)  # outlet edited
                        subidx0_out_lst.append(subidxs_out[idx1])
                        subidxs_out[idx1] = subidx_out1
                        # print('ds - outlet', idx1)
                    # update LATERAL connections
                    for k in ks:  # @4C loop lateral connections
                        idx0 = idxs_us0[k]
                        idx0_edit = idx0 in idx_out_lst
                        if idx0_edit:
                            continue  # @4C
                        subidx_ds0 = subidxs_ds0[k]
                        subidx = subidxs_out[idx0]
                        idx_ds0 = idx0
                        path = list()
                        # connect lateral to next downstream subgrid outlet
                        while True:  # 4D
                            # next downstream subgrid cell index
                            # NOTE use internal indices
                            subidx1 = subidxs_dense[subidxs_ds[subidxs_sparse[subidx]]]
                            idx_ds = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                            outlet = subidx1 == subidxs_out[idx_ds]
                            pit = subidx1 == subidx
                            idx_ds_edit = idx_ds0 in idx_out_lst
                            if outlet or pit:
                                # if ds direction and outlet unchanged, don't
                                # create bottleneck.
                                idx_ds0_edit = (
                                    idx0 in idx0_lst or nextidx[idx0] in idx_out_lst
                                )
                                # at outlet or at pit
                                ind8 = in_d8(idx0, idx_ds, ncol)
                                if (not ind8 and idx_ds0_edit) or (not outlet and pit):
                                    # outside d8 neighbors
                                    nextiter = True
                                    in_bottleneck = nextidx[idx0] in bottleneck
                                    if not in_bottleneck:
                                        bottleneck.append(nextidx[idx0])
                                        # print('bottleneck ', bottleneck)
                                elif ind8 and nextidx[idx0] != idx_ds:
                                    idx_ds0_lst.append(nextidx[idx0])
                                    idx0_lst.append(idx0)
                                    idx_ds_lst.append(idx_ds)
                                    nextidx[idx0] = idx_ds  # update
                                    # print('lats', idx0, idx_ds)
                                break  # @4D
                            # new cell AND
                            # passed unchanged donwstream outlet AND
                            elif (
                                idx_ds0 != idx_ds
                                and idx_ds0 != idx0
                                and subidx_ds0 in path
                                and not idx_ds_edit
                                and in_d8(idx0, idx_ds0, ncol)
                            ):
                                # at new cell / potential outlet
                                # set ds neighbor and relocate outlet IF
                                # the original outlet has zero upstream cells
                                # and the next downstream outlet is unchanged
                                # no old us neighbors
                                us_old0 = idxs_us[idxs_sparse[idx_ds0], 0] == _mv
                                # new us neighbors
                                us_new = idx_ds0 in idx_ds_lst
                                # next sugbrid outlet from
                                _, idx_ds00, outlet0 = next_outlet(
                                    subidx,
                                    subidxs_dense,
                                    subidxs_ds,
                                    subidxs_sparse,
                                    subidxs_out,
                                    subncol,
                                    cellsize,
                                    ncol,
                                )
                                idx_ds00_edit = idx_ds00 in idx_out_lst
                                # zero upstream cells from original outlet AND
                                # and unchanged next downstream outlet (not pit)
                                if (
                                    us_old0
                                    and not us_new
                                    and outlet0
                                    and not idx_ds00_edit
                                    and in_d8(idx_ds0, idx_ds00, ncol)
                                ):
                                    # update original next downstream cell
                                    if nextidx[idx0] != idx_ds0:
                                        idx_ds0_lst.append(nextidx[idx0])
                                        idx0_lst.append(idx0)
                                        idx_ds_lst.append(idx_ds0)
                                        nextidx[idx0] = idx_ds0
                                        # print('lat1', idx0, idx_ds0)
                                    if nextidx[idx_ds0] != idx_ds00:
                                        idx_ds0_lst.append(nextidx[idx_ds0])
                                        idx0_lst.append(idx_ds0)
                                        idx_ds_lst.append(idx_ds00)
                                        nextidx[idx_ds0] = idx_ds00
                                        # print('lat2', idx_ds0, idx_ds00)
                                    if subidx != subidxs_out[idx_ds0]:
                                        # outlet edited
                                        idx_out_lst.append(idx_ds0)
                                        subidx0_out_lst.append(subidxs_out[idx_ds0])
                                        subidxs_out[idx_ds0] = subidx
                                        # print('lat - outlet', idx_ds0)
                                    break  # @4D
                            # next iter @4D
                            path.append(subidx1)
                            subidx = subidx1
                            idx_ds0 = idx_ds
                    # next iter @4A
                    idx0 = idx1
                    j0 = j + 1
                # drop upstream lateral connections if past the connection
                # outlet which has not been edited
                elif not nextiter and lats:
                    for k in ks:  # @4E loop laterals with upstream connections
                        idx_ds0 = nextidx[idxs_us0[k]]
                        lat_ds = idx_ds0 in idxs_lst[j:]
                        lat_edit = idx_ds0 in idx_out_lst
                        if not lat_ds and not lat_edit:
                            k0 = k
                            # print('lats - skip', idxs_us0[k])
                        else:
                            break  # @4E

                # unroll edits
                if nextiter:
                    for i in range(len(idx0_lst)):
                        nextidx[idx0_lst[-1 - i]] = idx_ds0_lst[-1 - i]
                    for i in range(len(idx_out_lst)):
                        subidxs_out[idx_out_lst[i]] = subidx0_out_lst[i]

        # if next downstream in idx_out_lst we've created a loop
        loop = nextidx[idx1] in idx_out_lst
        if nextiter or loop:
            # print('failed -', idx00)
            if loop:
                # unroll edits
                for i in range(len(idx0_lst)):
                    nextidx[idx0_lst[-1 - i]] = idx_ds0_lst[-1 - i]
                for i in range(len(idx_out_lst)):
                    subidxs_out[idx_out_lst[i]] = subidx0_out_lst[i]
            continue  # @0A

        # next iter @A0
        for i in range(len(idx_ds_lst)):
            idx0, idx_ds, idx_ds0 = idx0_lst[i], idx_ds_lst[i], idx_ds0_lst[i]
            idxs_ds, idxs_us = update_idxs_ds_us(
                idx0, idx_ds, idx_ds0, idxs_ds, idxs_us, idxs_sparse
            )

    return nextidx, subidxs_out


def com(
    subidxs_ds,
    subidxs_dense,
    subuparea,
    subshape,
    cellsize,
    iter2=True,
    min_stream_len=1,
):
    """Returns the upscaled next downstream index based on the 
    connecting outlets method (COM).

    Parameters
    ----------
    subidxs_ds : ndarray of int
        internal highres indices of downstream cells
    subidxs_dense : ndarray of int
        highres raster indices of vaild cells
    subuparea : ndarray
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    cellsize : int
        size of lowres cell measured in higres cells
    iter2 : bool
        second iteration to improve subgrid connections
    min_stream_len : int
        minimum length (pixels) between outlet and previous confluence
        a confluence is determined by at least a doubling of upstream area

    Returns
    -------
    lowres indices of next downstream
        ndarray of int
    subgrid indices of outlet cells
        ndarray of int
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow = int(np.ceil(subnrow / cellsize))
    ncol = int(np.ceil(subncol / cellsize))
    shape = nrow, ncol
    # STEP 1
    # get representative cells
    subidxs_rep = eam_repcell(
        subidxs_ds, subidxs_dense, subuparea, subshape, shape, cellsize
    )
    # get subgrid outlet cells
    subidxs_out = com_outlets(
        subidxs_rep,
        subidxs_ds,
        subidxs_dense,
        subuparea,
        subshape,
        shape,
        cellsize,
        min_stream_len=min_stream_len,
    )
    # get next downstream lowres index
    nextidx, idxs_fix = com_nextidx(
        subidxs_out, subidxs_ds, subidxs_dense, subshape, shape, cellsize
    )
    # STEP 2 try fixing invalid subgrid connections
    if iter2:
        nextidx, subidxs_out = com_nextidx_iter2(
            nextidx,
            subidxs_out,
            idxs_fix,
            subidxs_ds,
            subidxs_dense,
            subuparea,
            subshape,
            shape,
            cellsize,
        )
    return nextidx.reshape(shape), subidxs_out.reshape(shape)

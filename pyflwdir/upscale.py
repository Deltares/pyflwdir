# -*- coding: utf-8 -*-
"""Methods for upscaling high res flow direction data to lower resolutions."""

from numba import njit
import numpy as np

from pyflwdir import core

_mv = core._mv

__all__ = []

# naming convention
# row,  col,    index,  width
# ----------------------------------------------
# r,    c,      idx,    ncol     -> lowres cells
# subr, subc,   subidx, subncol  -> highres cells
# ri,   ci,     ii,     cellsize -> location within lowres cell


#### GENERIC CONVENIENCE FUNCTIONS ####
@njit
def subidx_2_idx(subidx, subncol, cellsize, ncol):
    """Returns the lowres index <idx> of highres cell index <subidx>."""
    r = (subidx // subncol) // cellsize
    c = (subidx % subncol) // cellsize
    return r * ncol + c


@njit
def in_d8(idx0, idx_ds, ncol):
    """Returns True if inside 3x3 (current and 8 neighboring) cells."""
    cond1 = abs(idx_ds - idx0) <= 1  # west, east
    cond2 = abs(idx_ds - idx0 - ncol) <= 1  # south
    cond3 = abs(idx_ds - idx0 + ncol) <= 1  # north
    return cond1 or cond2 or cond3


#### DOUBLE MAXIMUM METHOD ####


@njit
def cell_edge(subidx, subncol, cellsize):
    """Returns True if highres cell <subidx> is on edge of lowres cell"""
    ri = (subidx // subncol) % cellsize
    ci = (subidx % subncol) % cellsize
    return ri == 0 or ci == 0 or ri + 1 == cellsize or ci + 1 == cellsize


@njit
def outlet_cells(idx, subidxs_ds, ncol, subncol, cellsize):
    """Returns subgrid cells at the edge of a lowres cells with the next downstream
    subgrid cell outside of that lowress cell."""
    subidxs = []
    subnrow = subidxs_ds.size / subncol
    r_ul = (idx // ncol) * cellsize
    c_ul = (idx % ncol) * cellsize
    r_ll = ((idx // ncol) + 1) * cellsize - 1
    c_ur = ((idx % ncol) + 1) * cellsize - 1
    subidx_ul = r_ul * subncol + c_ul
    subidx_ll = r_ll * subncol + c_ul
    subidx_ur = r_ul * subncol + c_ur
    for i in range(min(cellsize, subncol - c_ul)):
        subidx = subidx_ul + i
        if subidx_2_idx(subidxs_ds[subidx], subncol, cellsize, ncol) != idx:
            subidxs.append(subidx)
        if r_ll < subnrow:
            subidx = subidx_ll + i
            if subidx_2_idx(subidxs_ds[subidx], subncol, cellsize, ncol) != idx:
                subidxs.append(subidx)
    for i in range(1, min(cellsize - 1, subnrow - r_ul)):
        subidx = subidx_ul + i * subncol
        if subidx_2_idx(subidxs_ds[subidx], subncol, cellsize, ncol) != idx:
            subidxs.append(subidx)
        if c_ur < subncol:
            subidx = subidx_ur + i * subncol
            if subidx_2_idx(subidxs_ds[subidx], subncol, cellsize, ncol) != idx:
                subidxs.append(subidx)
    for subidx in subidxs:
        if not cell_edge(subidx, subncol, cellsize):
            print(idx, subidx)
            assert False
    return subidxs


@njit
def map_celledge(subidxs_ds, subshape, cellsize, mv=_mv):
    """Returns a map with ones on highres cells of lowres cell edges"""
    subncol = subshape[1]
    # allocate output array
    edges = np.full(subidxs_ds.size, -1, dtype=np.int8)
    # loop over valid indices
    for subidx in range(subidxs_ds.size):
        if subidxs_ds[subidx] == mv:
            continue
        if cell_edge(subidx, subncol, cellsize):
            edges[subidx] = np.int8(1)
        else:
            edges[subidx] = np.int8(0)
    return edges


@njit
def dmm_exitcell(subidxs_ds, subuparea, subshape, shape, cellsize, mv=_mv):
    """Returns exit highres cell indices of lowres cells according to the 
    double maximum method (DMM). 
    
    Parameters
    ----------
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subuparea : 1D-array of int
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
        highres indices of representative cells
    """
    _, subncol = subshape
    nrow, ncol = shape
    # allocate output
    subidxs_rep = np.full(nrow * ncol, mv, dtype=subidxs_ds.dtype)
    uparea = np.zeros(nrow * ncol, dtype=subuparea.dtype)
    # loop over valid indices
    for subidx in range(subidxs_ds.size):
        subidx_ds = subidxs_ds[subidx]
        if subidx_ds == mv:
            continue
        # NOTE including pits in the edge area is different from the original
        ispit = subidx_ds == subidx
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
def dmm_nextidx(subidxs_rep, subidxs_ds, subshape, shape, cellsize, mv=_mv):
    """Returns next downstream lowres index by tracing a representative cell 
    to where it leaves a buffered area around the lowres cell according to the 
    double maximum method (DMM). 
    
    Parameters
    ----------
    subidxs_rep : 1D-array of int
        highres linear indices of representative cells
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    1D-array of int
        lowres linear indices of next downstream cell
    """
    _, subncol = subshape
    nrow, ncol = shape
    R = cellsize / 2
    # allocate output
    idxs_ds = np.full(nrow * ncol, mv, dtype=subidxs_ds.dtype)
    # loop over rep cell indices
    for idx0 in range(subidxs_rep.size):
        subidx = subidxs_rep[idx0]
        idx = idx0
        if subidx == mv:
            continue
        # highres coordinates at center of offset lowres cell
        dr = (subidx // subncol) % cellsize // R
        dc = (subidx % subncol) % cellsize // R
        subr0 = (idx0 // ncol + dr) * cellsize - 0.5
        subc0 = (idx0 % ncol + dc) * cellsize - 0.5
        while True:
            # next downstream highres cell index
            subidx1 = subidxs_ds[subidx]
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


def dmm(subidxs_ds, subuparea, subshape, cellsize, mv=_mv):
    """Returns the upscaled next downstream index based on the 
    double maximum method (DMM) [1].

    ...[1] Olivera F, Lear M S, Famiglietti J S and Asante K 2002 
    "Extracting low-resolution river networks from high-resolution digital 
    elevation models" Water Resour. Res. 38 13-1-13–8 
    Online: http://doi.wiley.com/10.1029/2001WR000726
    
    Parameters
    ----------
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subuparea : 1D-array of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres linear indices of next downstream 
        1D-array of int
    highres linear indices of representative cells
        1D-array of int
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow = int(np.ceil(subnrow / cellsize))
    ncol = int(np.ceil(subncol / cellsize))
    shape = nrow, ncol
    # get outlet cells
    subidxs_out = dmm_exitcell(subidxs_ds, subuparea, subshape, shape, cellsize, mv)
    # get next downstream lowres index
    idxs_ds = dmm_nextidx(subidxs_out, subidxs_ds, subshape, shape, cellsize, mv)
    return idxs_ds, subidxs_out, shape


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
def map_effare(subidxs_ds, subshape, cellsize, mv=_mv):
    """Returns a map with ones on highres cells of lowres effective area."""
    subncol = subshape[1]
    # allocate output
    effare = np.full(subidxs_ds.size, -1, dtype=np.int8)
    # loop over valid indices
    for subidx in range(subidxs_ds.size):
        if subidxs_ds[subidx] == mv:
            continue
        if effective_area(subidx, subncol, cellsize):
            effare[subidx] = np.int8(1)
        else:
            effare[subidx] = np.int8(0)
    return effare


@njit
def eam_repcell(subidxs_ds, subuparea, subshape, shape, cellsize, mv=_mv):
    """Returns representative highres cell indices of lowres cells
    according to the effective area method. 
    
    Parameters
    ----------
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subuparea : 1D-array of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    highres representative cell indices : 1D-array with size shape[0]*shape[1]
    """
    _, subncol = subshape
    nrow, ncol = shape
    # allocate output
    subidxs_rep = np.full(nrow * ncol, mv, dtype=subidxs_ds.dtype)
    uparea = np.zeros(nrow * ncol, dtype=subuparea.dtype)
    # loop over valid indices
    for subidx in range(subidxs_ds.size):
        subidx_ds = subidxs_ds[subidx]
        if subidx_ds == mv:
            continue
        # NOTE including pits is different from the original EAM
        ispit = subidx_ds == subidx
        eff_area = effective_area(subidx, subncol, cellsize)
        # check upstream area if cell ispit or at effective area
        if ispit or eff_area:
            idx = subidx_2_idx(subidx, subncol, cellsize, ncol)
            upa0 = uparea[idx]
            upa = subuparea[subidx]
            # cell with largest upstream area is representative cell
            if upa > upa0:
                uparea[idx] = upa
                subidxs_rep[idx] = subidx
    return subidxs_rep


@njit
def eam_nextidx(subidxs_rep, subidxs_ds, subshape, shape, cellsize, mv=_mv):
    """Returns next downstream lowres index by tracing a representative cell to 
    the next downstream effective area according to the effective area method. 
    
    Parameters
    ----------
    subidxs_rep : 1D-array of int
        highres linear indices of representative subgird cells
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres linear indices of next downstream cell : 1D-array    
    """
    _, subncol = subshape
    nrow, ncol = shape
    # allocate output
    idxs_ds = np.full(nrow * ncol, mv, dtype=subidxs_ds.dtype)
    # loop over rep cell indices
    for idx0 in range(subidxs_rep.size):
        subidx = subidxs_rep[idx0]
        if subidx == mv:
            continue
        while True:
            # next downstream highres cell index
            subidx1 = subidxs_ds[subidx]
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


def eam(subidxs_ds, subuparea, subshape, cellsize, mv=_mv):
    """Returns the upscaled next downstream index based on the 
    effective area method (EAM) [1].

    ...[1] Yamazaki D, Masutomi Y, Oki T and Kanae S 2008 
    "An Improved Upscaling Method to Construct a Global River Map" APHW
    
    Parameters
    ----------
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subuparea : 1D-array of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres linear indices of next downstream
        1D-array of int
    highres linear indices of representative cells
        1D-array of int
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow = int(np.ceil(subnrow / cellsize))
    ncol = int(np.ceil(subncol / cellsize))
    shape = nrow, ncol
    # get representative cells
    subidxs_rep = eam_repcell(subidxs_ds, subuparea, subshape, shape, cellsize, mv)
    # get next downstream lowres index
    idxs_ds = eam_nextidx(subidxs_rep, subidxs_ds, subshape, shape, cellsize, mv)
    return idxs_ds, subidxs_rep, shape


#### CONNECTING OUTLETS SCALING METHOD ####
@njit
def com_outlets(
    subidxs_rep, subidxs_ds, subuparea, subshape, shape, cellsize, mv=_mv,
):
    """Returns highres outlet cell indices of lowres cells which are located
    at the edge of the lowres cell downstream of the representative cell
    according to the connecting outlets method (COM). 
    
    Parameters
    ----------
    subidxs_rep : 1D-array of int
        highres linear indices of representative cells with size shape[0]*shape[1]
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subuparea : 1D-array of int
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    raster with highres linear indices of outlets
        1D-array of int
    """
    _, subncol = subshape
    nrow, ncol = shape
    # allocate output
    subidxs_out = np.full(nrow * ncol, mv, dtype=subidxs_ds.dtype)
    # loop over rep cell indices
    for idx0 in range(subidxs_rep.size):
        subidx = subidxs_rep[idx0]
        if subidx == mv:
            continue
        while True:
            # next downstream highres cell index
            subidx1 = subidxs_ds[subidx]
            # at outlet if next highres cell is in next lowres cell
            outlet = idx0 != subidx_2_idx(subidx1, subncol, cellsize, ncol)
            pit = subidx1 == subidx
            if outlet or pit:
                break
            # next iter
            subidx = subidx1
        subidxs_out[idx0] = subidx
    return subidxs_out


@njit
def com_nextidx(subidxs_out, subidxs_ds, subshape, shape, cellsize, mv=_mv):
    """Returns next downstream lowres index according to connecting outlets 
    method (COM). Every outlet highres cell is traced to the next downstream 
    highres outlet cell. If this lays outside d8, we fallback to the next 
    downstream effective area.
    
    Parameters
    ----------
    subidxs_out : 1D-array of int
        highres linear indices of outlet cells with size shape[0]*shape[1]
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subshape : tuple of int
        highres raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres linear indices of next downstream, disconnected cells 
        Tuple of 1D-array    
    """
    _, subncol = subshape
    nrow, ncol = shape
    # allocate output
    idxs_ds = np.full(nrow * ncol, mv, dtype=subidxs_ds.dtype)
    idxs_fix_lst = list()
    # loop over outlet cell indices
    for idx0 in range(subidxs_out.size):
        subidx = subidxs_out[idx0]
        if subidx == mv:
            continue
        subidx_ds = mv
        while True:
            # next downstream highres cell index
            subidx1 = subidxs_ds[subidx]
            idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
            if subidxs_out[idx1] == subidx1 or subidx1 == subidx:
                # at outlet or pit
                if not in_d8(idx0, idx1, ncol):  # outside d8 neighbors
                    # flag index, use first pass ea
                    idxs_fix_lst.append(idx0)
                else:
                    subidx_ds = subidx1
                    # incorrect ds connection where pit
                    if subidxs_out[idx1] != subidx1:
                        idxs_fix_lst.append(idx0)  # flag index
                break
            if subidx_ds == mv and effective_area(subidx1, subncol, cellsize):
                subidx_ds = subidx1  # first pass effective area
            # next iter
            subidx = subidx1
        # assert subidx_ds != mv
        idxs_ds[idx0] = subidx_2_idx(subidx_ds, subncol, cellsize, ncol)
    return idxs_ds, np.array(idxs_fix_lst, dtype=subidxs_ds.dtype)


@njit
def next_outlet(
    subidx, subidxs_ds, subidxs_out, subncol, cellsize, ncol,
):
    """Returns lowres and highres indices of next outlet"""
    while True:
        # next downstream highres cell index
        subidx1 = subidxs_ds[subidx]
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
    idxs_fix,
    idxs_ds,
    subidxs_out,
    subidxs_ds,
    subuparea,
    subshape,
    shape,
    cellsize,
    mv=_mv,
):
    """Second iteration to relocate subgrid outlet cells in order to connect the
    subgrid outlets of disconnected cells.
    
    Parameters
    ----------
    idxs_fix : 1D-array of int
        lowres linear indices of cells which are disconnected in highres
    idxs_ds : 1D-array of int
        lowres linear indices of next downstream cell
    subidxs_out : 1D-array of int
        highres linear indices of outlet cells with size shape[0]*shape[1]
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subuparea : 1D-array of int
        highres flattened upstream area array
    subshape : tuple of int
        highres (highres) raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres linear indices of next downstream 
        1D-array of int
    highres linear indices of outlet cells
        1D-array of int  
    """
    _, subncol = subshape
    _, ncol = shape
    # loop over unconnected cells from up to downstream
    seq = np.argsort(subuparea[subidxs_out[idxs_fix]])
    for i0 in seq:  # @0A lowres fix index loop
        nextiter = False
        idx00 = idxs_fix[i0]
        # print(idx00)

        # STEP 1: get downstream path with highres outlet indices
        idxs_lst = list()
        subidxs_lst = list()
        connected = False
        # read outlet index and move into the next lowres cell to initialize
        idx_ds0 = idxs_ds[idx00]  # original next downstream cell
        subidx = subidxs_ds[subidxs_out[idx00]]  # next highres cell
        idx0 = subidx_2_idx(subidx, subncol, cellsize, ncol)
        while True:  # @1A while noy connected to original downstream cell
            # while True:  # @1B highres loop - while not at outlet
            subidx1 = subidxs_ds[subidx]
            idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
            pit = subidx1 == subidx
            if pit or idx0 != idx1:  # check pit or outlet
                # connected if:
                # - next downstream lowres cell not in path &
                # - current highres outlet cell same as original next outlet
                if pit:
                    connected = True
                elif subidx == subidxs_out[idx_ds0]:
                    if idx_ds0 in idxs_lst:
                        pass
                    else:
                        connected = True
                # check if valid cell
                if idxs_ds[idx0] != mv:
                    subidxs_lst.append(subidx)  # append highres outlet index
                    idxs_lst.append(idx0)  # append lowres index
                # if at original outlet cell of idx0 -> update idx_ds0
                if subidx == subidxs_out[idx0]:
                    idx_ds0 = idxs_ds[idx0]
                idx0 = idx1
            if connected:  # with original ds highres cell
                break  # @1A
            # next iter @1A
            subidx = subidx1
        if connected and subidx == subidxs_out[idxs_ds[idx00]]:
            # connection at first outlet -> already fixed
            continue  # @0A
        elif not connected:
            continue  # @0A

        # STEP 2: find original upstream connections
        idxs_us_lst = list()
        idxs_ds0 = np.unique(np.array(idxs_lst, dtype=idxs_fix.dtype))
        for idx_ds in idxs_ds0:  # @2A lowres us connections loop
            for idx0 in core._upstream_d8_idx(idx_ds, idxs_ds, shape):
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
            subidx = subidxs_ds[subidx]
            idx = idx0
            ii = 0
            while True and ii <= 10:  # @3B find connecting outlet in subidxs_lst
                subidx1 = subidxs_ds[subidx]
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
        idxs_us_conn = np.array(idxs_us_conn_lst, dtype=idxs_ds.dtype)
        seq1 = np.argsort(idxs_us_conn)
        idxs_us0 = np.array(idxs_us_lst, dtype=idxs_ds.dtype)[seq1]
        subidxs_ds0 = subidxs_out[idxs_ds[idxs_us0]]
        idxs_us_conn1 = np.array(idxs_us_conn_lst1, dtype=idxs_ds.dtype)[seq1]
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
                if subidx_out1 == subidxs_ds[subidx_out1] and len(idx_out_lst) == 0:
                    idxs_pit = np.where(subidxs_out == subidx_out1)[0]
                    if idxs_pit.size == 1 and in_d8(idx0, idxs_pit[0], ncol):
                        # previous (smaller) branch already claimed pit
                        if idxs_ds[idx0] != idxs_pit[0]:
                            idx_ds0_lst.append(idxs_ds[idx0])
                            idx0_lst.append(idx0)
                            idx_ds_lst.append(idxs_pit[0])
                            idxs_ds[idx0] = idxs_pit[0]
                            # print('pit - connect', idx0, idxs_pit[0])
                        break  # @4A
                    elif idxs_pit.size == 0:
                        # set pit and move outlet to neighboring cell
                        # NOTE pit outlet outside lowres cell !
                        if idxs_ds[idx0] != idx0:
                            idx_ds0_lst.append(idxs_ds[idx0])
                            idx0_lst.append(idx0)
                            idx_ds_lst.append(idx0)
                            idxs_ds[idx0] = idx0
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
                # if next highres outlet exists nextd8 is False -> force update
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
                    if idxs_ds[idx0] != idx1:
                        idx_ds0_lst.append(idxs_ds[idx0])
                        idx0_lst.append(idx0)
                        idx_ds_lst.append(idx1)
                        idxs_ds[idx0] = idx1
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
                        # connect lateral to next downstream highres outlet
                        while True:  # 4D
                            # next downstream highres cell index
                            subidx1 = subidxs_ds[subidx]
                            idx_ds = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                            outlet = subidx1 == subidxs_out[idx_ds]
                            pit = subidx1 == subidx
                            idx_ds_edit = idx_ds0 in idx_out_lst
                            if outlet or pit:
                                # if ds direction and outlet unchanged, don't
                                # create bottleneck.
                                idx_ds0_edit = (
                                    idx0 in idx0_lst or idxs_ds[idx0] in idx_out_lst
                                )
                                # at outlet or at pit
                                ind8 = in_d8(idx0, idx_ds, ncol)
                                if (not ind8 and idx_ds0_edit) or (not outlet and pit):
                                    # outside d8 neighbors
                                    nextiter = True
                                    in_bottleneck = idxs_ds[idx0] in bottleneck
                                    if not in_bottleneck:
                                        bottleneck.append(idxs_ds[idx0])
                                        # print('bottleneck ', bottleneck)
                                elif ind8 and idxs_ds[idx0] != idx_ds:
                                    idx_ds0_lst.append(idxs_ds[idx0])
                                    idx0_lst.append(idx0)
                                    idx_ds_lst.append(idx_ds)
                                    idxs_ds[idx0] = idx_ds  # update
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
                                idx_us0 = core._upstream_d8_idx(idx_ds0, idxs_ds, shape)
                                # next sugbrid outlet from
                                _, idx_ds00, outlet0 = next_outlet(
                                    subidx,
                                    subidxs_ds,
                                    subidxs_out,
                                    subncol,
                                    cellsize,
                                    ncol,
                                )
                                idx_ds00_edit = idx_ds00 in idx_out_lst
                                # zero upstream cells from original outlet AND
                                # and unchanged next downstream outlet (not pit)
                                if (
                                    idx_us0.size == 0
                                    # and not us_new
                                    and outlet0
                                    and not idx_ds00_edit
                                    and idx_ds0 != idx_ds00
                                    and in_d8(idx_ds0, idx_ds00, ncol)
                                ):
                                    # update original next downstream cell
                                    if idxs_ds[idx0] != idx_ds0:
                                        idx_ds0_lst.append(idxs_ds[idx0])
                                        idx0_lst.append(idx0)
                                        idx_ds_lst.append(idx_ds0)
                                        idxs_ds[idx0] = idx_ds0
                                        # print('lat1', idx0, idx_ds0)
                                    if idxs_ds[idx_ds0] != idx_ds00:
                                        idx_ds0_lst.append(idxs_ds[idx_ds0])
                                        idx0_lst.append(idx_ds0)
                                        idx_ds_lst.append(idx_ds00)
                                        idxs_ds[idx_ds0] = idx_ds00
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
                        idx_ds0 = idxs_ds[idxs_us0[k]]
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
                        idxs_ds[idx0_lst[-1 - i]] = idx_ds0_lst[-1 - i]
                    for i in range(len(idx_out_lst)):
                        subidxs_out[idx_out_lst[i]] = subidx0_out_lst[i]

        # if next downstream in idx_out_lst we've created a loop
        loop = idxs_ds[idx1] in idx_out_lst
        if loop:
            # unroll edits
            nextiter = True
            for i in range(len(idx0_lst)):
                idxs_ds[idx0_lst[-1 - i]] = idx_ds0_lst[-1 - i]
            for i in range(len(idx_out_lst)):
                subidxs_out[idx_out_lst[i]] = subidx0_out_lst[i]

    return idxs_ds, subidxs_out


@njit
def com_nextidx_iter3(
    idxs_fix,
    idxs_ds,
    subidxs_out,
    subidxs_ds,
    subbasins,
    subuparea,
    subshape,
    shape,
    cellsize,
    mv=_mv,
):
    """Third iteration to minimize the number of cells with an upstream area error when
    disconnected.
    
    Parameters
    ----------
    idxs_fix : 1D-array of int
        lowres linear indices of cells which are disconnected in highres
    idxs_ds : 1D-array of int
        lowres linear indices of next downstream cell
    subidxs_out : 1D-array of int
        highres linear indices of outlet cells with size shape[0]*shape[1]
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subbasins : 1D-array of int
        highres flattened basins map array
    subuparea : 1D-array of int
        highres flattened upstream area array
    subshape : tuple of int
        highres (highres) raster shape
    shape : tuple of int
        lowres raster shape
    cellsize : int
        size of lowres cell measured in higres cells

    Returns
    -------
    lowres linear indices of next downstream 
        1D-array of int
    highres linear indices of outlet cells
        1D-array of int  
    """
    _, subncol = subshape
    _, ncol = shape
    # binary array with outlets
    subn = subidxs_ds.size
    outlets = np.full(subn, mv, dtype=idxs_ds.dtype)
    for idx0 in range(idxs_ds.size):
        subidx = subidxs_out[idx0]
        if subidx == mv:
            continue
        outlets[subidx] = idx0
    # loop over cells
    for _ in range(5):
        idxs_fix_out = []
        for idx0 in idxs_fix:
            fixed = False
            subidx0 = subidxs_out[idx0]
            subidx_ds0 = subidxs_out[idxs_ds[idx0]]
            if subbasins is not None:
                bas0 = subbasins[subidx0]
            upa0 = subuparea[subidx0]
            # save path of cells with outlet pixel downstream of current outlet pixel
            idxs = []
            subidx = subidx0
            while True:
                subidx_ds = subidxs_ds[subidx]
                if outlets[subidx_ds] != mv:
                    if len(idxs) == 0 and subidx_ds == subidx_ds0:
                        fixed = True
                        break  # already fixed
                    idxs.append(outlets[subidx_ds])
                    if len(idxs) == 100:  # max 100 cells
                        break
                if subidx_ds == subidx:
                    break
                # next iter
                subidx = subidx_ds
            if fixed:
                continue
            if len(idxs) == 0 and subidx_ds != subidx_ds0 and subidx_ds == subidx:
                # pit -> reset outlet
                idxs_ds[idx0] = idx0
                subidxs_out[idx0] = subidx_ds
                outlets[subidx0] = mv
                outlets[subidx_ds] = idx0
                fixed = True
            elif len(idxs) > 0:
                # minimize total cells with upa error
                max_dist = 999999
                idxs_d8 = core._d8_idx(idx0, shape)
                if subbasins is not None:
                    bas_d8 = subbasins[subidxs_out[idxs_d8]]
                    same_bas = bas_d8 == bas0
                    idxs_d8 = idxs_d8[same_bas]
                for idx1 in idxs_d8:
                    subidx1 = subidxs_out[idx1]
                    upa1 = subuparea[subidx1]
                    idx = idx1
                    for j in range(max_dist + 1):
                        if idx in idxs:
                            d0 = idxs.index(idx) + j  # sum no of cells with error
                            if d0 < max_dist or (d0 == max_dist and upa1 > upa0):
                                idxs_ds[idx0] = idx1
                                max_dist = d0
                                upa0 = upa1
                                fixed = True
                            break
                        idx_ds = idxs_ds[idx]
                        if idx_ds == idx or idx_ds == idx0:  # break if pit or loop
                            break
                        # next iter
                        idx = idx_ds
                if subbasins is not None and not np.any(same_bas):
                    # no neighbors in same basin -> reset outlet cell
                    idxs_d8 = core._d8_idx(idx0, shape)
                    bas_unique = [b for b in np.unique(bas_d8)]
                    subidx_d8 = [subidxs_out[idx] for idx in idxs_d8]
                    upa_bas = np.zeros(len(bas_unique), dtype=subuparea.dtype)
                    subidxs_bas = np.zeros(len(bas_unique), dtype=subidxs_out.dtype)
                    idxs_ds_bas = np.zeros(len(bas_unique), dtype=idxs_ds.dtype)
                    subidxs = outlet_cells(idx0, subidxs_ds, ncol, subncol, cellsize)
                    for subidx0 in subidxs:
                        bas = subbasins[subidx0]
                        if bas in bas_unique:
                            upa = subuparea[subidx0]
                            ibas = bas_unique.index(bas)
                            if upa > upa_bas[ibas]:
                                subidx = subidx0
                                while True:
                                    subidx_ds = subidxs_ds[subidx]
                                    if outlets[subidx_ds] != mv or subidx == subidx_ds:
                                        if subidx_ds in subidx_d8:
                                            upa_bas[ibas] = upa
                                            idx_ds = idxs_d8[subidx_d8.index(subidx_ds)]
                                            idxs_ds_bas[ibas] = idx_ds
                                            subidxs_bas[ibas] = subidx0
                                        break
                                    # next iter
                                    subidx = subidx_ds
                    if np.any(upa_bas > 0):
                        ibas = np.argsort(upa_bas)[
                            -1
                        ]  # outlet with largest upstream area
                        idxs_ds[idx0] = idxs_ds_bas[ibas]
                        outlets[subidxs_out[idx0]] = mv
                        outlets[subidxs_bas[ibas]] = idx0
                        subidxs_out[idx0] = subidxs_bas[ibas]
                        fixed = True
            if not fixed:
                idxs_fix_out.append(idx0)
        if len(idxs_fix_out) == len(idxs_fix) or len(idxs_fix_out) == 0:
            break
        idxs_fix = np.array(idxs_fix_out, dtype=idxs_fix.dtype)

    return idxs_ds, subidxs_out


def com2(subidxs_ds, subuparea, subshape, cellsize, iter2=True, subbasins=None, mv=_mv):
    """Returns the upscaled next downstream index based on the 
    connecting outlets method (COM).

    Parameters
    ----------
    subidxs_ds : 1D-array of int
        highres linear indices of downstream cells
    subuparea : 1D-array
        highres flattened upstream area array
    subshape : tuple of int
        highres raster shape
    cellsize : int
        size of lowres cell measured in higres cells
    iter2 : bool
        second iteration to improve highres connections
    subbasins : 1D-array
        highres flattened basin map

    Returns
    -------
    lowres linear indices of next downstream
        1D-array of int
    highres linear indices of outlet cells
        1D-array of int
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow = int(np.ceil(subnrow / cellsize))
    ncol = int(np.ceil(subncol / cellsize))
    shape = nrow, ncol
    # STEP 1
    # get representative cells
    subidxs_rep = eam_repcell(
        subidxs_ds=subidxs_ds,
        subuparea=subuparea,
        subshape=subshape,
        shape=shape,
        cellsize=cellsize,
        mv=mv,
    )
    # get highres outlet cells
    subidxs_out = com_outlets(
        subidxs_rep=subidxs_rep,
        subidxs_ds=subidxs_ds,
        subuparea=subuparea,
        subshape=subshape,
        shape=shape,
        cellsize=cellsize,
        mv=mv,
    )
    # get next downstream lowres index
    idxs_ds, idxs_fix = com_nextidx(
        subidxs_out=subidxs_out,
        subidxs_ds=subidxs_ds,
        subshape=subshape,
        shape=shape,
        cellsize=cellsize,
        mv=mv,
    )
    if iter2:
        # sort from up to downstream
        seq = np.argsort(subuparea[subidxs_out[idxs_fix]])
        idxs_fix = idxs_fix[seq]
        idxs_ds, subidxs_out = com_nextidx_iter2(
            idxs_fix=idxs_fix,
            idxs_ds=idxs_ds,
            subidxs_out=subidxs_out,
            subidxs_ds=subidxs_ds,
            subuparea=subuparea,
            subshape=subshape,
            shape=shape,
            cellsize=cellsize,
            mv=mv,
        )
        idxs_ds, subidxs_out = com_nextidx_iter3(
            idxs_fix=idxs_fix,
            idxs_ds=idxs_ds,
            subidxs_out=subidxs_out,
            subidxs_ds=subidxs_ds,
            subbasins=subbasins,
            subuparea=subuparea,
            subshape=subshape,
            shape=shape,
            cellsize=cellsize,
            mv=mv,
        )
    return idxs_ds, subidxs_out, shape


def com(subidxs_ds, subuparea, subshape, cellsize, mv=_mv):
    return com2(subidxs_ds, subuparea, subshape, cellsize, iter2=False, mv=mv)


@njit
def connected(subidxs_out, idxs_ds, subidxs_ds, mv=_mv):
    """Returns an array with ones (zeros) if sugrid outlet/representative cells are 
    connected (disconnected) in D8, cells with missing values are set to -1. 
    
    Parameters
    ----------
    subidxs_out : 1D-array of int with same size as idxs_ds
        linear (highres) indices of unit catchment outlet cells
    idxs_ds : 1D-array of int with same size as subidxs_out
        linear lowres indices of next downstream cell
    subidxs_out, subidxs_ds : 1D-array of int
        linear highres indices of outlet, next downstream cells

    Returns
    -------
    1D-array of int
        ones where outlets are connected
    """
    assert subidxs_out.size == idxs_ds.size
    # binary array with outlets
    subn = subidxs_ds.size
    outlets = np.array([np.bool(0) for _ in range(subn)])
    for subidx in subidxs_out:
        if subidx == mv:
            continue
        outlets[subidx] = True
    # allocate output. intialize 'True' map
    n = idxs_ds.size
    connect_map = np.full(n, 1, np.uint8)
    # loop over outlet cell indices
    for idx0 in range(n):
        subidx = subidxs_out[idx0]
        idx_ds = idxs_ds[idx0]
        if idx_ds != mv and subidx != mv:
            while True:
                subidx1 = subidxs_ds[subidx]  # next downstream subgrid cell index
                if outlets[subidx1] or subidx1 == subidx:  # at outlet or at pit
                    if subidx1 != subidxs_out[idx_ds]:
                        connect_map[idx0] = np.uint8(0)
                    break
                # next iter
                subidx = subidx1
        else:
            connect_map[idx0] = np.uint8(-1)
    return connect_map

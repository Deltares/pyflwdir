# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np

# import flow direction definition
# from .utils import flwdir_check
from pyflwdir import core, core_nextidx
_mv = core._mv

# naming convention
# row,  col,    index,  width
# ----------------------------------------------
# r,    c,      idx,    ncol     -> lowres cells
# subr, subc,   subidx, subncol  -> subgrid/highres cells
# ri,   ci,     ii,     cellsize -> location within lowres cell 

#### GENERIC CONVENIENCE FUNCTIONS #### 
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
def in_d8(idx0, idx_ds, ncol):
    """Returns True if inside 3x3 (current and 8 neighboring) cells"""
    return abs(idx_ds-idx0) <= 1 or abs(idx_ds-idx0-ncol) <= 1 or abs(idx_ds-idx0+ncol) <= 1

@njit
def next_suboutlet(subidx, idx0, subidxs_internal, subidxs_ds, subidxs_valid, subncol, cellsize, ncol):
    """Returns the next downstream lowres subgrid outlet index of lowres cell"""
    path = list()
    while True:
        # next downstream subgrid cell index; complicated because of use internal indices 
        subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]] # next subgrid cell
        # at outlet if next subgrid cell is in next lowres cell
        outlet = idx0 != subidx_2_idx(subidx1, subncol, cellsize, ncol) 
        pit = subidx1 == subidx
        if outlet or pit: 
            break
        subidx = subidx1 # next iter
        path.append(subidx1)
    return subidx, np.array(path, dtype=subidxs_ds.dtype)

@njit
def cell_edge(subidx, subncol, cellsize):
    """Returns True if highres cell <subidx> is on edge of lowres cell"""
    ri = (subidx // subncol) % cellsize
    ci = (subidx %  subncol) % cellsize
    return ri == 0 or ci == 0 or ri+1 == cellsize or ci+1 == cellsize # desribes edge

@njit
def map_celledge(subidxs_ds, subidxs_valid, subshape, cellsize):
    """Returns a map with ones on subgrid cells of lowres cell edges"""
    surbnrow, subncol = subshape
    # allocate output and internal array
    edges = np.ones(surbnrow*subncol, dtype=np.int8)*-1
    # loop over valid indices
    for subidx in subidxs_valid:
        if cell_edge(subidx, subncol, cellsize):
            edges[subidx] = np.int8(1)
        else:
            edges[subidx] = np.int8(0)
    return edges.reshape(subshape)

@njit
def map_effare(subidxs_ds, subidxs_valid, subshape, cellsize):
    """Returns a map with ones on subgrid cells of lowres effective area"""
    surbnrow, subncol = subshape
    # allocate output and internal array
    effare = np.ones(surbnrow*subncol, dtype=np.int8)*-1
    # loop over valid indices
    for subidx in subidxs_valid:
        if effective_area(subidx, subncol, cellsize):
            effare[subidx] = np.int8(1)
        else:
            effare[subidx] = np.int8(0)
    return effare.reshape(subshape)

#### EFFECTIVE AREA METHOD ####

@njit
def effective_area(subidx, subncol, cellsize):
    """Returns True if highress cell <subidx> is inside the effective area"""
    R = cellsize / 2.
    offset = R-0.5 # lowres center at cellsize/2 - 0.5
    ri = abs((subidx // subncol) % cellsize - offset)
    ci = abs((subidx %  subncol) % cellsize - offset)
    return (ri**0.5 + ci**0.5 <= R**0.5) or (ri <= 0.5) or (ci <= 0.5) # describes effective area

@njit
def eam_repcell(subidxs_ds, subidxs_valid, subuparea, subshape, shape, cellsize):
    """Returns representative subgrid cell indices of lowres cells
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
    raster with subgrid representative cell indices : ndarray with size shape[0]*shape[1]
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
    lowres indices of next downstream cell : ndarray    
    """
    subnrow, subncol = subshape
    nrow, ncol = shape
    # internal indices
    n = subidxs_valid.size
    subidxs_internal = np.ones(subnrow*subncol, np.uint32)*_mv
    subidxs_internal[subidxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output 
    idxs_ds = np.ones(nrow*ncol, dtype=subidxs_ds.dtype)*_mv
    # loop over rep cell indices
    for idx0 in range(subidxs_out.size):
        subidx = subidxs_out[idx0]
        if subidx == _mv: 
            continue
        while True:
            # next downstream subgrid cell index; complicated because of use internal indices 
            subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]] 
            idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
            if subidx1 == subidx: # pit
                break
            elif idx1 != idx0 and effective_area(subidx1, subncol, cellsize): # in d8 effective area
                break
            # next iter
            subidx = subidx1 
        idxs_ds[idx0] = subidx_2_idx(subidx1, subncol, cellsize, ncol)
    return idxs_ds

def eam(subidxs_ds, subidxs_valid, subuparea, subshape, cellsize):
    """Returns the upscaled next downstream index based on the 
    effective area method.
    
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
    lowres indices of next downstream and subgrid indices of reprecentative cells : Tuple of ndarray
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow, ncol = int(np.ceil(subnrow/cellsize)), int(np.ceil(subncol/cellsize))
    shape = nrow, ncol
    # get representative cells    
    subidxs_rep = eam_repcell(subidxs_ds, subidxs_valid, subuparea, subshape, shape, cellsize)
    # get next downstream lowres index
    nextidx = eam_nextidx(subidxs_rep, subidxs_ds, subidxs_valid, subshape, shape, cellsize)
    return nextidx.reshape(shape), subidxs_rep

#### EXTENDED EFFECTIVE AREA METHOD ####
@njit
def eeam_outlets(subidxs_rep, subidxs_ds, subidxs_valid, subuparea, 
                    subshape, shape, cellsize, min_stream_len=0):
    """Returns subgrid outlet cell indices of lowres cells which are located
    at the edge of the lowres cell downstream of the representative cell
    according to the double effective area method. 

    NOTE: If <min_stream_len> is larger than zero, the outlet does not have to be
    the the lowres cells edge.
    
    Parameters
    ----------
    subidxs_rep : ndarray of int
        highres indices of representative cells with size shape[0]*shape[1]
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
    min_stream_len : int
        minimum length (pixels) between outlet and previous confluence
        a confluence is determined by at least a doubling of upstream area

    Returns
    -------
    raster with subgrid outlet indices : ndarray    
    """
    subnrow, subncol = subshape
    nrow, ncol = shape
    # internal indices
    n = subidxs_valid.size
    subidxs_internal = np.ones(subnrow*subncol, np.uint32)*core._mv
    subidxs_internal[subidxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output 
    subidxs_out = np.ones(nrow*ncol, dtype=subidxs_ds.dtype)*_mv
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
            # next downstream subgrid cell index; complicated because of use internal indices 
            subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]] # next subgrid cell
            if min_stream_len > 0:
                subupa1 = subuparea[subidx1]
                if subupa1 > 2*subupa: # confluence
                    subidx_prev_stream = subidx
                    stream_len = 0
            # at outlet if next subgrid cell is in next lowres cell
            outlet = idx0 != subidx_2_idx(subidx1, subncol, cellsize, ncol) 
            pit = subidx1 == subidx
            if outlet or pit:
                if min_stream_len > 0 and stream_len <= min_stream_len and subidx_prev_stream != _mv:
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
def eeam_nextidx(subidxs_out, subidxs_ds, subidxs_valid, subshape, shape, cellsize):
    """Returns next downstream lowres index according to EXTENDED effective area method. 
    Every outlet subgrid cell is traced to the next downstream subgrid outlet cell. If 
    this lays outside d8, we fallback to the next downstream effective area.
    
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
    lowres indices of next downstream and disconnected cells : Tuple of ndarray    
    """
    subnrow, subncol = subshape
    nrow, ncol = shape
    # internal indices
    n = subidxs_valid.size
    subidxs_internal = np.ones(subnrow*subncol, np.uint32)*_mv
    subidxs_internal[subidxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output 
    nextidx = np.ones(nrow*ncol, dtype=subidxs_ds.dtype)*_mv
    idxs_fix_lst = list()
    # loop over outlet cell indices
    for idx0 in range(subidxs_out.size):
        subidx = subidxs_out[idx0]
        if subidx == _mv:
            continue
        subidx_ds = _mv
        while True:
            # next downstream subgrid cell index; complicated because of use internal indices 
            subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]]
            idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
            if subidxs_out[idx1] == subidx1 or subidx1 == subidx: # at outlet or at pit
                if not in_d8(idx0, idx1, ncol): # outside d8 neighbors
                    idxs_fix_lst.append(np.uint32(idx0)) # flag index, use first pass ea
                else:
                    subidx_ds = subidx1
                    if subidxs_out[idx1] != subidx1: # incorrect ds connection where pit
                        idxs_fix_lst.append(np.uint32(idx0)) # flag index
                break
            if subidx_ds == _mv and effective_area(subidx1, subncol, cellsize):
                subidx_ds = subidx1 # first pass effective area
            # next iter
            subidx = subidx1 
        # assert subidx_ds != _mv
        nextidx[idx0] = subidx_2_idx(subidx_ds, subncol, cellsize, ncol)
    return nextidx, np.array(idxs_fix_lst, dtype=np.uint32)

@njit
def eeam_nextidx_iter2(
        nextidx, subidxs_out, idxs_fix,
        subidxs_ds, subidxs_valid,
        subuparea, subshape, shape, cellsize):
    """
    
    Parameters
    ----------
    nextidx : ndarray of int
        raster with next downstream lowres indices
    subidxs_out : ndarray of int
        subgrid indices of outlet cells with size shape[0]*shape[1]
    idxs_fix : ndarray of int
        lowres indices of cells which are disconnected in subgrid
    subidxs_ds : ndarray of int
        internal subgrid indices of downstream cells
        NOTE these are internal indices for valid cells only
    subidxs_valid : ndarray of int
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
    nrow, ncol = shape
    # parse nextidx
    idxs_valid, _, idxs_us, _  = core_nextidx.from_flwdir(nextidx.reshape(shape))
    # internal indices
    n = idxs_valid.size
    idxs_internal = np.ones(nrow*ncol, np.uint32)*_mv
    idxs_internal[idxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    subn = subidxs_valid.size
    subidxs_internal = np.ones(subnrow*subncol, np.uint32)*_mv
    subidxs_internal[subidxs_valid] = np.array([i for i in range(subn)], dtype=np.uint32)
    # allocate output
    nextidx1 = nextidx.copy()
    subidxs_out1 = subidxs_out.copy()
    idxs_fix_lst = list()
    # loop over unconnected cells from up to downstream
    upa_check = subuparea[subidxs_out[idxs_fix]]
    seq = np.argsort(upa_check)
    for i0 in seq: # @0A lowres fix index loop
        nextiter = False
        idx00 = idxs_fix[i0]
        # print(idx00)    

        # STEP 1: get downstream path with subgrid outlet indices 
        idxs_lst = list()
        subidxs_lst = list()
        connected = False
        # read outlet index and move into the next lowres cell to initialize
        subidx = subidxs_out1[idx00] 
        idx_ds0 = nextidx1[idx00]    # original next downstream cell
        subidx = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]]
        idx0 = subidx_2_idx(subidx, subncol, cellsize, ncol)
        while True: # @1A lowres loop - while noy connected to original downstream cell
            while True: # @1B subgrid loop - while not at outlet
                subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]]
                idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                pit = subidx1 == subidx
                if pit or idx0 != idx1: # check pit or outlet
                    # connected if:
                    # next downstream lowres cell not in path &
                    # current subgrid outlet cell is same as original next outlet
                    if pit:
                        connected = True
                    elif subidx == subidxs_out1[idx_ds0]:
                        if idx_ds0 in idxs_lst:
                            pass
                        else:
                            connected = True
                    subidxs_lst.append(subidx) # append subgrid outlet index
                    idxs_lst.append(idx0)      # append lowres index
                    # if at original outlet cell of idx0 -> update idx_ds0
                    if subidx == subidxs_out1[idx0]:
                        idx_ds0 = nextidx[idx0]
                    break # @1B
                # next iter @1B
                subidx = subidx1 
            if connected: # with original ds subgrid cell 
                break # @1A
            # next iter @1A
            idx0 = idx1
            subidx = subidx1
        if connected and subidx == subidxs_out1[nextidx1[idx00]]: # connection at first outlet -> already fixed
            continue # @0A
        elif not connected:
            idxs_fix_lst.append(idx00)
            continue # @0A

        # STEP 2: find original upstream connections 
        idxs_us_lst = list()
        idxs_ds0 = np.unique(np.array(idxs_lst, dtype=idxs_fix.dtype))
        for idx_ds in idxs_ds0: # @2A lowres us connections loop
            for i in idxs_us[idxs_internal[idx_ds],:]:
                if i == _mv:
                    break # @2A 
                idx0 = idxs_valid[i]
                # skip upstream nodes wich are on path of step 1
                if subidxs_out1[idx0] in subidxs_lst or idx0 == idx00: 
                    continue # @2A
                idxs_us_lst.append(idx0) # append lowres index of upstream connection

            
        # STEP 3: connect original upstream connections to outlets on path
        noutlets = len(subidxs_lst)
        idxs_us_conn_lst = list()  # first possible connection of lateral
        idxs_us_conn_lst1 = list() # last possible connection of lateral
        for i in range(len(idxs_us_lst)): # @3A lowres us connections loop
            idx0 = idxs_us_lst[i]
            subidx = subidxs_out1[idx0]
            connected = False
            j0, j1 = 0, 0
            # move into next cell to initialize
            subidx = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]]
            idx = idx0
            while True: # @3B subgrid loop to find connecting outlet in subidxs_lst
                subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]] 
                idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                if (subidx == subidx1 or idx != idx1): # if pit OR outlet
                    for j in range(j0, noutlets): # @3C check outlet loop
                        if subidxs_lst[j] == subidx: # connection on path found
                            if not connected:
                                j0, j1, connected = j, j, True
                            elif in_d8(idx0, idx, ncol):
                                j1 = j
                            break # @3C
                    if (j1+1 == noutlets) or subidx == subidx1:
                        break # @3C
                # next iter @3B
                subidx = subidx1
                idx = idx1
            if connected:
                idxs_us_conn_lst.append(j0)
                idxs_us_conn_lst1.append(j1)
            else:
                idxs_us_conn_lst.append(noutlets-1)
                idxs_us_conn_lst1.append(noutlets-1)        

        # STEP 4: connect the dots
        bottleneck = list()
        nbottlenecks = -1
        while len(bottleneck) > nbottlenecks:
            nextiter = False
            nbottlenecks = len(bottleneck)
            idxs_edit_lst = list()
            nextidx2 = nextidx1.copy()
            subidxs_out2 = subidxs_out1.copy()
            seq1 = np.argsort(np.array(idxs_us_conn_lst, dtype=nextidx.dtype)) # sort from up to downstream
            idxs_us0 = np.array(idxs_us_lst, dtype=nextidx.dtype)[seq1]
            idxs_us_conn = np.array(idxs_us_conn_lst, dtype=nextidx.dtype)[seq1]
            idxs_us_conn1 = np.array(idxs_us_conn_lst1, dtype=nextidx.dtype)[seq1]
            idx0 = idx00
            j0, k0 = 0, 0
            for j in range(noutlets): # @4A lowres connecting loop
                idx1 = idxs_lst[j]
                subidx_out1 = subidxs_lst[j]
                # check if not connected to ds pit
                pit = subidx_out1 == subidxs_valid[subidxs_ds[subidxs_internal[subidx_out1]]]
                if pit and len(idxs_edit_lst) == 0:
                    idxs_pit = np.where(subidxs_out2 == subidx_out1)[0]
                    if idxs_pit.size == 1: # previous (smaller) branch already claimed pit
                        nextidx2[idx0] = idxs_pit[0]
                    else:
                        nextidx2[idx0] = idx0
                        subidxs_out2[idx0] = subidx_out1 # NOTE pit outlet outside original cell
                    break # @4A
                # check if ds lowres cell already edited to avoid loops
                d8 = False if idx1 in idxs_edit_lst or idx1 in bottleneck else in_d8(idx0, idx1, ncol)
                # check lateral connections
                ks = np.where(np.logical_and(idxs_us_conn[k0:]>=j0, idxs_us_conn[k0:]<=j))[0]+k0
                lats = ks.size > 0
                nextlats = np.all(idxs_us_conn1[ks]>j) if lats else False
                # check if possible d8 connection downstream
                nextd8 = False
                for jj in range(j+1, noutlets):
                    if idxs_lst[jj] in idxs_edit_lst or idxs_lst[jj] in bottleneck:
                        continue
                    elif in_d8(idx0, idxs_lst[jj], ncol):
                        nextd8 = True
                    if subidxs_out2[idxs_lst[jj]] == subidxs_lst[jj]: # original outlet
                        break
                # if next subgird outlet exists nextd8 is False to force d8c
                nextd8 = nextd8 and subidxs_out2[idx1] != subidx_out1
                # print(idx0, idx1)
                # print(d8, nextd8, lats, nextlats)
                if (not d8 and not nextd8):
                    nextiter = True
                    break # @4A
                elif (not lats and nextd8) or (nextlats and nextd8):
                    continue
                # UPDATE CONNECTIONS
                if (d8 and  lats) or (d8 and not nextd8):
                    # update main connection
                    nextidx2[idx0] = idx1
                    subidxs_out2[idx1] = subidx_out1
                    idxs_edit_lst.append(idx1) # outlet edited
                    # print('ds', idx0, idx1)
                    # import pdb; pdb.set_trace()
                    # update lateral connections
                    for k in ks: # @4C loop lateral connections
                        idx0 = idxs_us0[k]
                        if idx0 in idxs_edit_lst:
                            continue # @4C, already edited
                        subidx = subidxs_out2[idx0]
                        idx_ds0 = idx0
                        path = list()
                        while True: # 4D connect lateral to next downstream subgrid outlet
                            # next downstream subgrid cell index; complicated because of use internal indices 
                            subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]]
                            idx_ds = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                            if subidx1 == subidxs_out2[idx_ds] or subidx1 == subidx: # at outlet or at pit
                                if not in_d8(idx0, idx_ds, ncol): # outside d8 neighbors
                                    nextiter = True
                                    if nextidx1[idx0] in bottleneck:
                                        pass
                                    else:
                                        bottleneck.append(nextidx1[idx0])
                                        # print('bottleneck ', bottleneck)
                                else:
                                    nextidx2[idx0] = idx_ds # update
                                    # print('lats', idx0, idx_ds)
                                    # import pdb; pdb.set_trace()
                                break # @4D
                            elif idx_ds0 != idx_ds and idx_ds0 != idx0: # lowres cell outlet
                                if idx_ds0 in idxs_edit_lst:
                                    pass
                                elif (idxs_us[idxs_internal[idx_ds0],0] == _mv and  # zero upstream cells
                                        subidxs_out1[nextidx2[idx0]] in path):      # original outlet
                                    nextidx2[idx0] = idx_ds0 # update
                                    nextidx2[idx_ds0] = nextidx1[nextidx1[idx0]] # original next downstream cell
                                    subidxs_out2[idx_ds0] = subidx
                                    idxs_edit_lst.append(idx_ds0) # outlet edited
                                    # print('lats - outlet', idx0, idx_ds0)
                                    # import pdb; pdb.set_trace()
                                    break # @4D
                            # next iter @4D
                            path.append(subidx1)
                            subidx = subidx1 
                            idx_ds0 = idx_ds
                    # next iter @4A
                    if nextiter:
                        break # @4A
                    idx0 = idx1 
                    j0 = j+1
                # drop upstream lateral connections if past the connection outlet which 
                # has not been edited
                if lats:
                    for k in ks: # @4E loop laterals with upstream connections
                        idx_ds0 = nextidx2[idxs_us0[k]]
                        lat_ds = idx_ds0 in idxs_lst[j:]
                        lat_edit = idx_ds0 in idxs_edit_lst
                        if not lat_ds and not lat_edit:
                            k0 = k
                        else:
                            break # @4E

        # if next downstream in idxs_edit_lst we've created a loop -> break.
        if nextiter or nextidx2[idx1] in idxs_edit_lst:
            idxs_fix_lst.append(idx00)
            continue # @0A
        
        # next iter @A0
        ndiff = np.sum(subidxs_out1 != subidxs_out2) + np.sum(nextidx1 != nextidx2)
        if ndiff > 0:
            _, idxs_ds, idxs_us, _ = core_nextidx.from_flwdir(nextidx2.reshape(shape))
            assert core.loop_indices(idxs_ds, idxs_us).size == 0 # loop at idx00
            # if core.loop_indices(idxs_ds, idxs_us).size > 0:
            #     print(idx00)
            #     import pdb; pdb.set_trace()
            nextidx1 = nextidx2
            subidxs_out1 = subidxs_out2

    return nextidx1, subidxs_out1, np.array(idxs_fix_lst, dtype=np.uint32)

def eeam(subidxs_ds, subidxs_valid, subuparea, subshape, cellsize, 
            iter2=True, min_stream_len=1):
    """Returns the upscaled next downstream index based on the 
    EXTENDED effective area method.

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
    iter2 : bool
        second iteration to improve subgrid connections
    min_stream_len : int
        minimum length (pixels) between outlet and previous confluence
        a confluence is determined by at least a doubling of upstream area

    Returns
    -------
    lowres indices of next downstream and subgrid indices of outlet cells : Tuple of ndarray
    """
    # calculate new size
    subnrow, subncol = subshape
    nrow, ncol = int(np.ceil(subnrow/cellsize)), int(np.ceil(subncol/cellsize))
    shape = nrow, ncol
    # get representative cells    
    subidxs_rep = eam_repcell(subidxs_ds, subidxs_valid, subuparea, subshape, shape, cellsize)
    # get subgrid outlet cells
    subidxs_out = eeam_outlets(subidxs_rep, subidxs_ds, subidxs_valid, subuparea, 
                                subshape, shape, cellsize, min_stream_len=min_stream_len)
    # get next downstream lowres index
    nextidx, idxs_fix = eeam_nextidx(subidxs_out, subidxs_ds, subidxs_valid, subshape, shape, cellsize)
    print(idxs_fix.size)
    # second iteration to improve subgrid connections
    if iter2:
        # idxs_fix = np.array([4263], dtype=np.uint32)
        nextidx, subidxs_out, idxs_fix = eeam_nextidx_iter2(
                nextidx, subidxs_out, idxs_fix,
                subidxs_ds, subidxs_valid,
                subuparea, subshape, shape, cellsize)
        print(idxs_fix.size)

    return nextidx.reshape(shape), subidxs_out, idxs_fix
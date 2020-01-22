# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np

# import flow direction definition
# from .utils import flwdir_check
from pyflwdir import core, core_nextidx
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

# @njit
# def not_d8(idx0, subidx, subncol, cellsize, ncol):
#     """Returns True if outside 3x3 (current and 8 neighboring) cells"""
#     idx_ds = subidx_2_idx(subidx, subncol, cellsize, ncol)
#     return not in_d8(idx0, idx_ds, ncol)

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
    """Returns True if highress cell <subidx> is on edge of lowres cell"""
    ri = (subidx // subncol) % cellsize
    ci = (subidx %  subncol) % cellsize
    return ri == 0 or ci == 0 or ri+1 == cellsize or ci+1 == cellsize # desribes edge

@njit
def map_celledge(subidxs_ds, subidxs_valid, subshape, cellsize):
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

# NOTE search until out of D8 -> does not seem to improve results !
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
    subidxs_internal = np.ones(subnrow*subncol, np.uint32)*core._mv
    subidxs_internal[subidxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output 
    subidxs_out = np.ones(nrow*ncol, dtype=subidxs_ds.dtype)*_mv
    # loop over rep cell indices
    for idx0 in range(subidxs_rep.size):
        subidx = subidxs_rep[idx0]
        if subidx == _mv: 
            continue
        while True:
            # next downstream subgrid cell index; complicated because of use internal indices 
            subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]] # next subgrid cell
            # at outlet if next subgrid cell is in next lowres cell
            outlet = idx0 != subidx_2_idx(subidx1, subncol, cellsize, ncol) 
            pit = subidx1 == subidx
            if outlet or pit: 
                break
            subidx = subidx1 # next iter
        subidxs_out[idx0] = subidx
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
    subidxs_internal = np.ones(subnrow*subncol, np.uint32)*_mv
    subidxs_internal[subidxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output 
    idxs_ds = np.ones(nrow*ncol, dtype=subidxs_ds.dtype)*_mv
    # loop over outlet cell indices
    for idx0 in range(subidxs_out.size):
        subidx = subidxs_out[idx0]
        if subidx == _mv: 
            continue
        # move one subgrid cell downstream into next lowres cell
        subidx = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]] 
        while not effective_area(subidx, subncol, cellsize):
            # next downstream subgrid cell index; complicated because of use internal indices 
            subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]] 
            if subidx1 == subidx: # pit
                break
            subidx = subidx1 # next iter
        idxs_ds[idx0] = subidx_2_idx(subidx, subncol, cellsize, ncol)
    return idxs_ds

def eam(subidxs_ds, subidxs_valid, subuparea, subshape, cellsize):
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
    return nextidx.reshape(shape), subidxs_out

#### EXTENDED EFFECTIVE AREA METHOD ####
@njit
def eeam_nextidx(subidxs_out, subidxs_ds, subidxs_valid, subshape, shape, cellsize):
    """Returns next downstream lowres index by tracing a representative cell to the 
    next downstream effective area according to the EXTENDED effective area method. 
    
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
    subidxs_internal = np.ones(subnrow*subncol, np.uint32)*_mv
    subidxs_internal[subidxs_valid] = np.array([i for i in range(n)], dtype=np.uint32)
    # allocate output 
    idxs_ds = np.ones(nrow*ncol, dtype=subidxs_ds.dtype)*_mv
    idxs_fix_lst = list()
    # loop over outlet cell indices
    for idx0 in range(subidxs_out.size):
        subidx = subidxs_out[idx0]
        subidx_ds = _mv
        if subidx == _mv:
            continue
        while True:
            # next downstream subgrid cell index; complicated because of use internal indices 
            subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]]
            idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
            if subidxs_out[idx1] == subidx1 or subidx1 == subidx: # at outlet or at pit
                if not in_d8(idx0, idx1, ncol): # outside d8 neighbors
                    idxs_fix_lst.append(np.uint32(idx0)) # flag index, use first pass ea
                else:
                    subidx_ds = subidx1
                break
            if subidx_ds == _mv and effective_area(subidx1, subncol, cellsize):
                subidx_ds = subidx1 # first pass effective area
            subidx = subidx1 # next iter
        # assert subidx_ds != _mv
        idxs_ds[idx0] = subidx_2_idx(subidx_ds, subncol, cellsize, ncol)
    return idxs_ds, np.array(idxs_fix_lst, dtype=np.uint32)

# @njit
def eeam_nextidx_iter2(
        nextidx, subidxs_out, idxs_fix,       # 
        subidxs_ds, subidxs_valid,            # subgrid high res flwdir arrays
        subuparea, subshape, shape, cellsize):
    """
    
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
        if idx00 == 117662:
            print(idx00)
            import pdb; pdb.set_trace()

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
        idxs_us_conn_lst = list()
        for i in range(len(idxs_us_lst)): # @3A lowres us connections loop
            idx0 = idxs_us_lst[i]
            subidx = subidxs_out1[idx0]
            connected = False
            # move into next cell to initialize
            subidx = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]]
            while True: # @3B subgrid loop to find connecting outlet in subidxs_lst
                subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]] 
                idx1 = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                pit = subidx == subidx1
                if pit or (idx0 != idx1 and cell_edge(subidx, subncol, cellsize)): # check outlet
                    for j in range(noutlets): # @3C check outlet loop
                        if subidxs_lst[j] == subidx: # ds connection found
                            idxs_us_conn_lst.append(j) # append position in <subidxs_lst> of subgrid outlet 
                            connected = True
                            break # @3C
                        elif pit:
                            break # @3C
                if connected or pit:
                    break # @3B
                # next iter @3B
                subidx = subidx1
            if idx00 == 117662 and not connected:
                print(idx0)
                import pdb; pdb.set_trace()
            if not connected:
                nextiter = True
                break # @3A
        if nextiter:  
            idxs_fix_lst.append(idx00)
            continue # @0A
        

        # STEP 4: connect the dots
        idxs_edit_lst = list()
        nextidx2 = nextidx1.copy()
        subidxs_out2 = subidxs_out1.copy()
        seq1 = np.argsort(np.array(idxs_us_conn_lst, dtype=nextidx.dtype)) # sort from up to downstream
        idxs_us0 = np.array(idxs_us_lst, dtype=nextidx.dtype)[seq1]
        idxs_us_conn = np.array(idxs_us_conn_lst, dtype=nextidx.dtype)[seq1]
        idx0 = idx00
        j0 = 0
        for j in range(noutlets): # @4A lowres connecting loop
            idx1 = idxs_lst[j]
            subidx_out1 = subidxs_lst[j]
            # check if ds lowres cell already edited to avoid loops
            if idx1 in idxs_edit_lst:
                d8 = False
            else:
                d8 = in_d8(idx0, idx1, ncol)
            # check if lateral connections
            ks = np.where(np.logical_and(idxs_us_conn>=j0, idxs_us_conn<=j))[0]
            lats = ks.size > 0
            # next node \w lateral
            j1 = idxs_us_conn[idxs_us_conn>=j0][0] if np.any(idxs_us_conn>=j0) else j
            # check if possible d8 connection between current node and next node \w lateral
            # if next subgird outlet exists nextd8 is False
            nextd8conn = False
            for jj in range(j+1, min(max(j1+1, j+2), noutlets)):
                if idxs_lst[jj] in idxs_edit_lst:
                    continue
                elif in_d8(idx0, idxs_lst[jj], ncol):
                    nextd8conn = True
                    break
            nextd8 = (
                j+1<noutlets and 
                subidxs_out2[idx1] != subidx_out1 and 
                nextd8conn
            )
            if idx00 == 117662:
                print(idx00)
                import pdb; pdb.set_trace()
            if not d8 and not nextd8:
                nextiter = True
                break # @4A
            elif not lats and nextd8:
                continue
            elif lats and nextd8:
                idx_ds = idxs_lst[j+1]
                connected = True
                for k in ks: # @4B loop lateral connections
                    if not in_d8(idxs_us0[k], idx_ds, ncol):
                        connected = False
                        break # @4B
                if not d8 and not connected:
                    nextiter = True
                    break # @4A
                elif connected:
                    continue # @4A
            # UPDATE CONNECTIONS
            if (lats and d8) or (d8 and not nextd8):
                # update main connection
                nextidx2[idx0] = idx1
                subidxs_out2[idx1] = subidx_out1
                idxs_edit_lst.append(idx1) # outlet edited
                # update lateral connections
                for k in ks: # @4C loop lateral connections
                    idx0 = idxs_us0[k]
                    if idx0 in idxs_edit_lst:
                        continue # @4C, already edited
                    subidx = subidxs_out2[idx0]
                    while True: # 4D connect lateral to next downstream subgrid outlet
                        # next downstream subgrid cell index; complicated because of use internal indices 
                        subidx1 = subidxs_valid[subidxs_ds[subidxs_internal[subidx]]]
                        idx_ds = subidx_2_idx(subidx1, subncol, cellsize, ncol)
                        if subidxs_out2[idx_ds] == subidx1 or subidx1 == subidx: # at outlet or at pit
                            if not in_d8(idx0, idx_ds, ncol): # outside d8 neighbors
                                nextiter = True
                            else:
                                # TODO: double check if this creates loops
                                nextidx2[idx0] = idx_ds # update
                            break # @4D
                        subidx = subidx1 # next iter
                # next iter @4A
                if nextiter:
                    break # @4A
                idx0 = idx1 
                j0 = j+1
            else:
                assert False, 'should not happen' # TODO remove after testing
        # if next downstream in idxs_edit_lst we've created a loop -> break.
        if nextiter or nextidx2[idx1] in idxs_edit_lst:
            idxs_fix_lst.append(idx00)
            continue # @0A
        
        # next iter @A0
        ndiff = np.sum(subidxs_out1 != subidxs_out2) + np.sum(nextidx1 != nextidx2)
        if ndiff > 0:
            _, idxs_ds, idxs_us, _ = core_nextidx.from_flwdir(nextidx2.reshape(shape))
            # if core.loop_indices(idxs_ds, idxs_us).size > 0:
            #     print(idx00)
            #     import pdb; pdb.set_trace()
            #     break # @A0
            assert core.loop_indices(idxs_ds, idxs_us).size == 0
            nextidx1 = nextidx2
            subidxs_out1 = subidxs_out2


    return nextidx1, subidxs_out1, np.array(idxs_fix_lst, dtype=np.uint32)

def eeam(subidxs_ds, subidxs_valid, subuparea, subshape, cellsize, iter2=True):
    """Returns the upscaled flow direction network based on the EXTENDED effective area method
    
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
    nextidx, idxs_fix = eeam_nextidx(subidxs_out, subidxs_ds, subidxs_valid, subshape, shape, cellsize)
    print(idxs_fix.size)
    # print(222585 in idxs_fix) # TODO check for this index how it works
    # idxs_fix = np.array([222585], dtype=np.uint32)
    if iter2:
        nextidx, subidxs_out, idxs_fix = eeam_nextidx_iter2(
                nextidx, subidxs_out, idxs_fix,       # 
                subidxs_ds, subidxs_valid,            # subgrid high res flwdir arrays
                subuparea, subshape, shape, cellsize)
        print(idxs_fix.size)

    # import pdb; pdb.set_trace()
    return nextidx.reshape(shape), subidxs_out, idxs_fix
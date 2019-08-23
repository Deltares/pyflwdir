# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np

# import flow direction definition
from .utils import flwdir_check
from .core import fd
_nodata = fd._nodata
_pits = fd._pits 
_ds = fd._ds

@njit
def subidx_2_idx(subidx, sub_ncol, scale_ratio):
    """Return the index of the lowres cell in which the subgrid cell <subidx> is located"""
    ncol = int(sub_ncol//scale_ratio)
    r = (subidx // sub_ncol) // scale_ratio
    c = (subidx %  sub_ncol) // scale_ratio
    return r * ncol + c

@njit
def cellidx_2_subidx(tilesubidx, tileidx, sub_ncol, scale_ratio):
    """Returns the global subgrid index based on the local tile subgrid index"""
    ncol = int(sub_ncol/scale_ratio)
    r = tileidx // ncol * scale_ratio + tilesubidx // scale_ratio
    c = tileidx %  ncol * scale_ratio + tilesubidx %  scale_ratio
    return r * sub_ncol + c

@njit
def _rep_cell(idx0, flwdir_flat, uparea_flat, shape, scale_ratio):
    """Returns the representative cell of a subgrid cell as well as all subgrid
    cells inside the effective area. The effective area is decribed by eq1; while
    the representative cell is the cell with the largest upstream area inside the 
    effective area. Note that the effective area is designed in a way that the 
    representative cell, when followed downstream, always connects to the effective 
    area of a neighboring cell in d8. 

    eq1) eff_area = (i**0.5 + j**0.5 <= R**0.5) or (i <= 0.5) or (j <= 0.5)
    """
    R = scale_ratio/2.
    rr = np.abs(np.arange(-R+0.5, R, 1.))
    sub_ncol = shape[1]
    upa0 = uparea_flat[0]*0
    idx, subidx, subidx0 = np.int64(-1), np.int64(-1), np.int64(-1) 
    repcells = []
    for i in rr:
        for j in rr:
            idx += 1
            subidx = cellidx_2_subidx(idx, idx0, sub_ncol, scale_ratio)
            ispit = fd.ispit(flwdir_flat[subidx])
            eff_area = (i**0.5 + j**0.5 <= R**0.5) or (i <= 0.5) or (j <= 0.5) # describes effective area
            if ispit or eff_area: 
                upa = uparea_flat[subidx]
                if eff_area:  
                    repcells.append(subidx)
                if upa > upa0:
                    upa0 = upa
                    subidx0 = subidx
    
    if subidx0 != -1 and fd.ispit(flwdir_flat[subidx0]) and np.all(np.array(repcells)!=subidx0):
        repcells.append(subidx0)
    elif subidx0 == -1:
        # select center cell if no rep cell found
        idx = int(round((scale_ratio**2)/2.))
        subidx0 = cellidx_2_subidx(idx, idx0, sub_ncol, scale_ratio)
    return subidx0, np.array(repcells)

@njit
def _outlet(idx0, subidx, flwdir_flat, shape, scale_ratio):
    """Returns outlet of current cell <idx0> downstream of subgrid cell <subidx>
    as well as the path leading to it.  Here an outlet is the last subgrid cell 
    before leaving the lowres cell"""
    subidx_out, subidx_ds = np.int64(-1), np.int64(-1)
    sub_ncol = shape[1]
    idx = idx0
    streamcells = [subidx]
    while True:
        subidx_ds = fd.ds_index(subidx, flwdir_flat, shape) # move downstream
        # outside domain or at pit/mouth
        if subidx_ds == -1 or subidx_ds == subidx: 
            if idx==idx0:
                subidx_out = subidx
            break
        idx_ds = subidx_2_idx(subidx_ds, sub_ncol, scale_ratio)
        # center lowres cell
        if idx_ds==idx0 and idx==idx0: 
            streamcells.append(subidx_ds)
        # moving into nb cell 
        elif idx==idx0 and idx_ds!=idx0: 
            subidx_out = subidx
            break
        # next iteration
        idx = idx_ds
        subidx = subidx_ds
    return subidx_out, np.array(streamcells)

@njit
def _dd(idx0, subidx, flwdir_flat, effare_flat, shape, shape_lr, scale_ratio, extended=True):
    """Returns the drainage direction <dd> for cell <idx0> as well
    The drainage direction is determined by following the subgrid outlet cell <subidx> downstream to where 
     a) if extended == False: it meets the next effective area. This requirement is always met.
     b) if extended == True: it meets the next subgrid outlet cell. If no outlet cell is found within 
        the 8 neighbors, the last effective area determines the drainage direction 
    """
    sub_ncol = shape[1]
    ncol = int(sub_ncol/scale_ratio)
    dd = _pits[0]
    idx_ds, idx_ds0 = np.int64(-1), np.int64(-1)
    i = 0
    while True:
        subidx_ds = fd.ds_index(subidx, flwdir_flat, shape) # move downstream
        if subidx_ds == -1: # outside domain
            break 
        # at pit/mouth
        elif subidx_ds == subidx:
            dd = flwdir_flat[subidx]
            break
        idx_ds = subidx_2_idx(subidx_ds, sub_ncol, scale_ratio)
        # center lowres cell
        if idx_ds == idx0:
            pass
        # outside 3x3 lowres neighbors
        elif abs(idx_ds-idx0) > 1 and abs(idx_ds-idx0-ncol) > 1 and abs(idx_ds-idx0+ncol) > 1:
            # return last cell inside 3x3 window
            subidx_ds = subidx 
            idx_ds = subidx_2_idx(subidx_ds, sub_ncol, scale_ratio)
            break
        # in neighboring lowres cell
        else:  
            # check if in subgrid hits outlet / eff area
            flag = effare_flat[subidx_ds] # 1 eff area; 2 stream to outlet; 3 outlet
            if  flag >= np.uint8(1):
                if idx_ds0 != idx_ds:
                    dd = fd.idx_to_dd(idx0, idx_ds, shape_lr)
                    idx_ds0 = idx_ds
                    # assert dd != _nodata
                if not extended or flag >= np.uint8(2):
                    break
                i += 1
        # next iteration
        subidx = subidx_ds
    return dd, subidx_ds, idx_ds

@njit
def _ds_effare_then_outlet(subidx0, scale_ratio, flwdir_flat, effare_flat, shape):
    """Returns the outlet subgrid cell downstream from the next downstream effective area.
    The <effare_flat> map should have zeros in all cells upstream from the effective area and 
    larger or equal to one inside and downstream of the effective area.
    """
    sub_ncol = shape[1]
    subidx_out = -1
    idx = -1
    # @ upstream point
    subidx = subidx0
    ea = np.uint8(0)
    while ea == np.uint(0): # requires highres grid with effective area cell == 1
        subidx_ds = fd.ds_index(subidx, flwdir_flat, shape) # move downstream
        if subidx_ds == -1 or subidx_ds == subidx:
            break 
        ea = effare_flat[subidx_ds]
        subidx = subidx_ds
    # @ downstream effective area
    if ea != np.uint8(0):
        idx_ds = subidx_2_idx(subidx, sub_ncol, scale_ratio) # calc lowres idx of subgrid cell
        subidx_out = _outlet(idx_ds, subidx, flwdir_flat, shape, scale_ratio)[0] # move to outlet
        # @ downstream outlet
    return idx_ds, subidx_out

@njit
def _us_main_branch(subidx0, idx0, scale_ratio, flwdir_flat, uparea_flat, shape, upa_min=0):
    """Returns the subgrid branch with the largest upstream area upstream from <subidx0> inside lowres cell <idx0> """
    sub_ncol = shape[1]
    subidx = subidx0
    idx = idx0
    upa_branch = upa_min
    subidx_branch = -1
    while idx == idx0: # break if main us in other lowres cell
        subidxs_us = fd.us_indices(subidx, flwdir_flat, shape)
        upa0 = upa_min
        subidx0 = -1
        for subidx_us in subidxs_us:
            upa = uparea_flat[subidx_us]
            # update next main upstream
            if upa > upa0:
                # update branch subidx if previous main us larger uparea and same lowres cell
                if upa0 > upa_branch and subidx_2_idx(subidx0, sub_ncol, scale_ratio) == idx0:
                    upa_branch = upa0
                    subidx_branch = subidx0
                upa0 = upa
                subidx0 = subidx_us
            # update branch subidx if larger uparea and same lowres cell
            elif upa > upa_branch and subidx_2_idx(subidx_us, sub_ncol, scale_ratio) == idx0:
                upa_branch = upa
                subidx_branch = subidx_us
        if subidxs_us.size == 0: 
            break # no more upstream cells
        subidx = subidx0
        idx = subidx_2_idx(subidx0, sub_ncol, scale_ratio)
    return subidx_branch

@njit
def _ddext(idx0, flwdir_lr_flat, outlet_lr_flat, checkd_lr_flat, 
            flwdir_flat, uparea_flat, effare_flat, shape, shape_lr, scale_ratio):
    """Returns lists with changes in drainage directon and subgrid indices by in order to create a properly
    connected network. 

    Two cells are properly connected on a subgrid level if the outlet of one cell connects to the outlet
    of a downstream neighboring cell without passing other outlet cells.

    First 
    
    It fails if not all requirements can be fullfilled.
    """
    # outputs
    idx0_lst = []
    dd0_lst = []
    idx1_lst = []
    out1_lst = []
    # initialize
    idx = idx0
    subidx0 = outlet_lr_flat[idx0]
    idx_ds0 = fd.ds_index(idx0, flwdir_lr_flat, shape_lr)
    idx2_lst = [] # check upstream branches
    idx3_lst = [] # avoid subidx "in between" outlets 
    i = 0
    success = False
    # move downstream and connect main branch
    while not success and i < 5:
        # find next downstream outlet
        # in a cell which is not already been modified
        dsout = False
        for _ in range(3):
            subidx_out0 = outlet_lr_flat[idx_ds0] 
            if idx_ds0 in idx1_lst:
                subidx_out0 = out1_lst[np.where(np.array(idx1_lst)==idx_ds0)[0][0]]
            idx_ds, subidx_out = _ds_effare_then_outlet(subidx0, scale_ratio, flwdir_flat, effare_flat, shape)
            if idx_ds not in idx0_lst and subidx_out0 not in idx3_lst and checkd_lr_flat[idx_ds] == 0:
                dd = fd.idx_to_dd(idx, idx_ds, shape_lr)
                if fd.ispit(dd) == False and dd != _nodata:
                    dsout = True
                    break
            idx_ds0 = fd.ds_index(idx_ds, flwdir_lr_flat, shape_lr)
            subidx0 = subidx_out
            idx3_lst.append(subidx_out)
        if not dsout: break # unsuccessfull
        # append dd changes
        idx0_lst.append(idx)
        dd0_lst.append(dd)
        # append upstream branches of new ds cell for later check
        idxs_us0 = fd.us_indices(idx_ds, flwdir_lr_flat, shape_lr)
        for idx_us in idxs_us0:
            if idx_us != idx and idx_us not in idx2_lst:
                idx2_lst.append(idx_us)
        # # append upastream branches of original ds cell for later check
        if idx_ds != idx_ds0:
            idxs_us0 = fd.us_indices(idx_ds0, flwdir_lr_flat, shape_lr)
            for idx_us in idxs_us0:
                if idx_us != idx and idx_us not in idx2_lst:
                    idx2_lst.append(idx_us)
        # successfull changed main ds path if next outlet unchanged
        if subidx_out0 == subidx_out: 
            success = True
            break
        # append subidx changes
        out1_lst.append(subidx_out)
        idx1_lst.append(idx_ds)
        # next iter
        idx = idx_ds
        subidx0 = subidx_out
        idx_ds0 = fd.ds_index(idx, flwdir_lr_flat, shape_lr)
        i += 1

    # check upsteam branches
    if success:
        for idx in idx2_lst:
            if idx in idx0_lst: continue
            subidx0 = outlet_lr_flat[idx]
            # replace outlet idx if changed
            if idx in idx1_lst:
                subidx0 = out1_lst[np.where(np.array(idx1_lst)==idx)[0][0]]
                assert subidx0 not in idx3_lst
            # find largest upstream subgrid branch
            if subidx0 in idx3_lst:
                subidx_branch = _us_main_branch(subidx0, idx, scale_ratio, flwdir_flat, uparea_flat, shape)    
                out1_lst.append(subidx_branch)
                idx1_lst.append(idx)            
            # check if branch connects to next downstream outlet
            else:
                dsout = False
                for _ in range(5):
                    # move to outlet of next ds lowres cell
                    idx_ds = idx
                    while idx_ds == idx:
                        subidx_ds = fd.ds_index(subidx0, flwdir_flat, shape)
                        if subidx_ds == -1 or subidx_ds == subidx0: break
                        subidx0 = subidx_ds
                        idx_ds = subidx_2_idx(subidx_ds, shape[1], scale_ratio)
                    if idx_ds == idx: break
                    subidx_out = _outlet(idx_ds, subidx_ds, flwdir_flat, shape, scale_ratio)[0] # move to outlet
                    # get the cell outlets
                    if idx_ds not in idx1_lst:
                        subidx_out0 = outlet_lr_flat[idx_ds]
                    else: 
                        subidx_out0 = out1_lst[np.where(np.array(idx1_lst)==idx_ds)[0][0]]
                    # next ds outlet found if cell outlet == ds outlet
                    if subidx_out0 == subidx_out:
                        dsout = True
                        break
                    subidx0 = subidx_out
                # check if connects in d8
                if dsout:
                    dd = fd.idx_to_dd(idx, idx_ds, shape_lr)
                    if dd == _nodata: 
                        success = False
                        break
                    elif dd != flwdir_lr_flat[idx]:
                        idx0_lst.append(idx)
                        dd0_lst.append(dd)
                else:
                    success = False
                    break
    return success, np.array(idx0_lst), np.array(dd0_lst), np.array(idx1_lst), np.array(out1_lst)



@njit
def _ddplus(idx0, flwdir_lr_flat, outlet_lr_flat, flwdir_flat, uparea_flat, effare_flat, 
            shape, shape_lr, scale_ratio, upa_min=0.5):
    idx_us_main = idx0
    dd_new = _nodata
    subidx_out = outlet_lr_flat[idx0]
    subidx_out_new = subidx_out
    subidx_ds, subidx_us = np.int64(-1), np.int64(-1)
    # check if valid upstream & downstream cells
    idxs_us = fd.us_indices(idx0, flwdir_lr_flat, shape_lr)
    idx_ds = fd.ds_index(idx0, flwdir_lr_flat, shape_lr)
    if idxs_us.size == 0 or idx_ds == idx0 or idx_ds < 0:
        return idx0, _nodata, subidx_out
    # check if main us connects to next ds in d8
    subidxs_out_us = outlet_lr_flat[idxs_us]
    maxi = np.argmax(uparea_flat[subidxs_out_us])
    idx_us_main = idxs_us[maxi]
    dd_new = fd.idx_to_dd(idx_us_main, idx_ds, shape_lr)
    assert not fd.ispit(dd_new)
    if dd_new == _nodata: # -1 if it does not connect in d8
        return idx0, _nodata, subidx_out
    # trace main stream downstream
    subidx_us = subidxs_out_us[maxi]
    mainstream = [subidx_us]
    while True:
        subidx_ds = fd.ds_index(subidx_us, flwdir_flat, shape)
        if subidx_ds == subidx_us or subidx_ds == -1:
            break
        mainstream.append(subidx_ds)
        if effare_flat[subidx_ds] == np.uint8(3):
            break
        subidx_us = subidx_ds # next iter
    if mainstream[-1] != subidx_out:
        return idx0, _nodata, subidx_out
    # find tributaries within max 3 cells from outlet
    n = len(mainstream)
    mainstream = mainstream[::-1]
    tributaries = []
    j = 0
    while True and j < n-1:
        subidx_ds = mainstream[j]
        subidx_us_main = mainstream[j+1]
        subidxs_us = fd.us_indices(subidx_ds, flwdir_flat, shape)
        for subidx_us in subidxs_us:
            if subidx_us == subidx_us_main:
                continue
            elif uparea_flat[subidx_us] > upa_min and subidx_2_idx(subidx_us, shape[1], scale_ratio) == idx0:
                tributaries.append(subidx_us)
        if effare_flat[subidx_us_main] != 2:
            break
        j += 1
    if len(tributaries) == 0:
        return idx0, _nodata, subidx_out
    if idxs_us.size > 1:
        # if upstream cells other than main upstream;
        # check if these connect to same tributary == new outlet cell
        sound = True
        for i in range(idxs_us.size):
            if i == maxi: continue
            subidx_us = subidxs_out_us[i]
            while sound:
                subidx_ds = fd.ds_index(subidx_us, flwdir_flat, shape)
                invalid = subidx_ds == subidx_us or subidx_ds == -1 or effare_flat[subidx_ds] == np.uint8(3)
                attrib = subidx_ds in tributaries
                if attrib and (subidx_ds == subidx_out_new or subidx_out_new == subidx_out):
                    subidx_out_new = subidx_ds
                    break
                elif attrib or invalid or (subidx_ds in mainstream):
                    sound = False
                    break
                subidx_us = subidx_ds # next iter
        if sound == False or subidx_out_new == subidx_out:
            return idx0, _nodata, subidx_out
    else:
        # if no upstream cells > new outlet is largest tributary
        maxi_trib = np.argmax(uparea_flat[np.array(tributaries)])
        subidx_out_new = tributaries[maxi_trib]
        
    return idx_us_main, dd_new, subidx_out_new

@njit
def _fix_pit_outlets(idx0, subidx0, subidx_pit, flwdir_flat, uparea_flat, effare_flat, 
                    shape, shape_lr, scale_ratio, upa_min):
    # deal with pits outside of lowres cells
    sub_ncol = shape[1]
    ncol = shape_lr[1]
    idx_pit = subidx_2_idx(subidx_pit, sub_ncol, scale_ratio) # low res index of pit
    assert idx0 != idx_pit
    dd = fd.idx_to_dd(idx0, idx_pit, shape_lr)
    # check if subidx0 on main river upstream from subidx_pit
    main = True
    subidx = subidx0
    upa = uparea_flat[subidx]
    while True:
        subidx_ds = fd.ds_index(subidx, flwdir_flat, shape) # move downstream
        upa_ds = uparea_flat[subidx_ds]
        main = (upa_ds / upa) < 2 # if larger than 2 we encounter a larger stream
        if subidx_ds == subidx_pit or main == False:
            break
        # next iter
        subidx = subidx_ds
        upa = upa_ds
    # if on main river move outlet, dd is pit
    if main:
        dd = flwdir_flat[subidx_pit]
        subidx0 = subidx_pit # NOTE: outlet outside of lowres cell!
    # else try to connect to main upstream outlet from pit
    else:
        subidx = subidx_ds
        while True:
            subidxs_us_main = fd.us_main_indices(subidx, flwdir_flat, uparea_flat, shape, upa_min=upa_min)[0]
            if subidxs_us_main.size == 0: # no outlet found
                break
            subidx_us_main = subidxs_us_main[0]
            assert subidx_us_main != subidx
            idx_us = subidx_2_idx(subidx_us_main, sub_ncol, scale_ratio)
            if abs(idx_us-idx0) > 1 and abs(idx_us-idx0-ncol) > 1 and abs(idx_us-idx0+ncol) > 1: 
                break
            elif effare_flat[subidx_us_main] == 3 and idx_us != idx0:  # 3 outlet
                dd = fd.idx_to_dd(idx0, idx_us, shape_lr) # change dd
                break
            # next iter
            subidx = subidx_us_main

    return dd, subidx0

@njit
def d8_scaling(scale_ratio, flwdir_flat, uparea_flat, shape, upa_min=2.0, extended=True):
    sub_nrow, sub_ncol = shape
    lr_nrow, lr_ncol = int(sub_nrow/scale_ratio), int(sub_ncol/scale_ratio)
    shape_lr = (lr_nrow, lr_ncol)
    size_lr = lr_nrow * lr_ncol
    # output cells
    outlet_lr_flat = np.ones(size_lr, dtype=np.int64)*-1
    flwdir_lr_flat = np.zeros(size_lr, dtype=flwdir_flat.dtype)
    checkd_lr_flat = np.zeros(size_lr, dtype=np.uint8) # to modify outlets only once in _ddext
    changd_lr_flat = np.zeros(size_lr, dtype=np.uint8) # to keep track of changes for debugging
    effare_flat = np.zeros(flwdir_flat.size, dtype=np.uint8)   
    idx_check = []
    upa_check = []
    # find outlet cells and save to highres and lowres grids
    for idx0 in range(outlet_lr_flat.size):
        subidx, subidxs_ea = _rep_cell(idx0, flwdir_flat, uparea_flat, shape, scale_ratio)
        subidx_out, subidxs_rc = _outlet(idx0, subidx, flwdir_flat, shape, scale_ratio)
        outlet_lr_flat[idx0] = subidx_out   # subgrid index of outlet point lowres grid cell
        # NOTE sequence is important
        effare_flat[subidxs_rc] = np.uint8(2) # 2 for representative stream cells (between representative cell and outlet)
        effare_flat[subidxs_ea] = np.uint8(1) # save effective area
        effare_flat[subidx_out] = np.uint8(3)


    for idx0 in range(flwdir_lr_flat.size):
        subidx = outlet_lr_flat[idx0]
        dd, subidx_out, idx_ds = _dd(idx0, subidx, flwdir_flat, effare_flat, shape, shape_lr, scale_ratio, extended=extended)
        if subidx_out != -1 and fd.ispit(dd) and not fd.ispit(flwdir_flat[subidx]):  # fix pit outlets ouside lowres cell
            dd, subidx_out = _fix_pit_outlets(
                idx0, subidx, subidx_out, flwdir_flat, uparea_flat, effare_flat, shape, shape_lr, scale_ratio, upa_min)
            if fd.ispit(dd):
                # assert fd.ispit(flwdir_flat[subidx_out])
                outlet_lr_flat[idx0] = subidx_out # change outlet. NOTE the outlet lies outside the lowres cell
        elif dd != _nodata and effare_flat[subidx_out] <= np.uint(1): # subgrid outlets might not be connected
            # check if not connected
            ea = np.uint8(1)
            while ea != np.uint8(3):
                subidx_ds = fd.ds_index(subidx, flwdir_flat, shape)
                if subidx_ds == -1 or subidx_ds == subidx: break
                ea = effare_flat[subidx_ds]
                subidx = subidx_ds
            idx_ds = subidx_2_idx(subidx, shape[1], scale_ratio)
            if  fd.idx_to_dd(idx0, idx_ds, shape_lr) != dd: # not connected
                idx_check.append(idx0)
                upa_check.append(uparea_flat[subidx])
                changd_lr_flat[idx0] = np.uint(1)
        flwdir_lr_flat[idx0] = dd

    assert flwdir_check(flwdir_lr_flat, shape_lr)[1] == False # no loops
    
    # second iteration to check misconnected
    if extended:
        seq = np.argsort(np.array(upa_check)) # from small to large
        for i in seq:
        # for i in range(len(idx_check)):
            idx0 = idx_check[i]
            if checkd_lr_flat[idx0] == 1: continue
            success, idxs_dd, dds, idxs_out, subidxs_out = _ddext(
                idx0, flwdir_lr_flat, outlet_lr_flat, checkd_lr_flat, 
                flwdir_flat, uparea_flat, effare_flat, shape, shape_lr, scale_ratio
            )
            if success:
                if idxs_dd.size > 0:
                    flwdir_lr_flat[idxs_dd] = dds
                if idxs_out.size > 0:
                    outlet_lr_flat[idxs_out] = subidxs_out
                    checkd_lr_flat[idxs_out] = np.uint8(1)
                changd_lr_flat[idx0] = np.uint(2)

    # second dd iteration to change outlets if near tributary
    if extended:
        for idx0 in range(flwdir_lr_flat.size):
            if flwdir_lr_flat[idx0] == _nodata: continue
            idx_us_main, dd_new, subidx_out_new = _ddplus(
                idx0, flwdir_lr_flat, outlet_lr_flat, flwdir_flat, uparea_flat, effare_flat, 
                shape, shape_lr, scale_ratio, upa_min=upa_min
            )
            if dd_new != _nodata:
                flwdir_lr_flat[idx_us_main] = dd_new
                outlet_lr_flat[idx0] = subidx_out_new
                if changd_lr_flat[idx0] == 0: 
                    changd_lr_flat[idx0] = np.uint(3)


    # assert outlet_lr_flat.size == np.unique(outlet_lr_flat).size
    assert flwdir_check(flwdir_lr_flat, shape_lr)[1] == False # no loops
    return flwdir_lr_flat.reshape(shape_lr), outlet_lr_flat.reshape(shape_lr), changd_lr_flat.reshape(shape_lr)


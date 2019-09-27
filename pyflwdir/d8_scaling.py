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
_ds = fd._ds.ravel()
_us = fd._us.ravel()


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
def _force_ds_outlet(idx0, subidx0, scale_ratio, flwdir_flat, effare_flat, shape, shape_lr, n):
    """Returns the nth outlet subgrid cell downstream from the next downstream effective area.
    The <effare_flat> map should have zeros in all cells upstream from the effective area and 
    larger or equal to one inside and downstream of the effective area.
    """
    sub_ncol = shape[1]
    subidx1 = -1
    idx_ds1 = -1
    dd1 = fd._nodata
    # @ upstream point
    subidx = subidx0
    path = [subidx]
    ea = np.uint8(0)
    i = 0
    idx_ds = -1
    if n > 0:
        while ea == np.uint8(0): # move to next effecitve area
            subidx_ds = fd.ds_index(subidx, flwdir_flat, shape) # move downstream
            if subidx_ds == -1 or subidx_ds == subidx:
                break 
            ea = effare_flat[subidx_ds]
            subidx = subidx_ds
            path.append(subidx)
        i += 1
    else:
        while idx_ds == idx0: # move out of current cell
            subidx_ds = fd.ds_index(subidx, flwdir_flat, shape) # move downstream
            idx_ds = subidx_2_idx(subidx, sub_ncol, scale_ratio)
            subidx = subidx_ds
            path.append(subidx)
        ea = np.uint8(1)
    # find nth downstream outlet from downstream eff area
    if ea != np.uint8(0):
        # @ move to next downstream outlet
        j = 0
        while j < 3:
            idx_ds = subidx_2_idx(subidx, sub_ncol, scale_ratio)
            subidx, path2 = _outlet(idx_ds, subidx, flwdir_flat, shape, scale_ratio)
            dd = fd.idx_to_dd(idx0, idx_ds, shape_lr)
            path.extend(path2)
            if dd != _nodata and fd.ispit(dd) == False:
                if i == n:
                    idx_ds1 = idx_ds
                    subidx1 = subidx
                    dd1 = dd
                    break
                i += 1
            else:
                j += 1
            subidx_ds = fd.ds_index(subidx, flwdir_flat, shape)
            if subidx_ds == subidx or subidx_ds == -1:
                break
            subidx = subidx_ds

    return idx_ds1, subidx1, dd1, path

@njit
def _ddext(idx0, flwdir_lr_flat, outlet_lr_flat, checkd_lr_flat, 
            flwdir_flat, uparea_flat, effare_flat, shape, shape_lr, scale_ratio):
    """"""
    n = 1
    success = False
    bottleneck = []
    nb = 0
    while True:
        # outputs
        idx0_lst = []
        dd0_lst = []
        idx1_lst = []
        out1_lst = []
        # initialize    
        idx_path = [] 
        path = []
        i = 0
        # force a new downstream outlet 
        idx1 = idx0
        subidx = outlet_lr_flat[idx1] 
        idx_ds1, subidx1, dd1, path1 = _force_ds_outlet(
            idx1, subidx, scale_ratio, flwdir_flat, effare_flat, shape, shape_lr, n
        )
        while not success and i < 5:
            if idx_ds1 == -1 or fd.ispit(flwdir_lr_flat[idx_ds1]): break #  or checkd_lr_flat[idx_ds1] == 1
            # get original ds outlet location
            idx_ds0 = fd.ds_index(idx1, flwdir_lr_flat, shape_lr)
            subidx0 = outlet_lr_flat[idx_ds0]
            if idx_ds0 in idx1_lst:
                subidx0 = out1_lst[np.where(np.array(idx1_lst)==idx_ds0)[0][0]]
            # append dd
            idx0_lst.append(idx1)
            dd0_lst.append(dd1)
            # append outlet changes
            if outlet_lr_flat[idx_ds1] != subidx1:
                out1_lst.append(subidx1)
                idx1_lst.append(idx_ds1)
            idx_path.append(idx_ds1)
            # append stream path
            path.extend(path1)
            # success if new ds outlet at original ds outlet
            if subidx1 == subidx0:
                success = True
                break
            # find next downtstream cell which is not on streampath or previously modified cell or pit
            idx1 = idx_ds1
            subidx = subidx1
            nn = 1
            while idx_ds1 in idx_path + bottleneck or fd.ispit(flwdir_lr_flat[idx_ds1]): #  or checkd_lr_flat[idx_ds1] == 1
                idx_ds1, subidx1, dd1, path1 = _force_ds_outlet(
                    idx1, subidx, scale_ratio, flwdir_flat, effare_flat, shape, shape_lr, nn
                )
                if idx_ds1 == -1: break 
                nn += 1
                # subidx = subidx1
            i += 1

        if success:
            # if original outlets on stream path, move to largest upstream branch
            branch = False
            idx1 = -1
            subidx1 = -1
            path = path[::-1] # down- to upstream !
            for i in range(len(path)):
                subidx = path[i] 
                idx = subidx_2_idx(subidx, shape[1], scale_ratio)
                # check for original outlet on flowpath. move outlet to largest branch in cell
                if idx != idx1:
                    if branch and subidx1 != -1:
                        out1_lst.append(subidx1)
                        idx1_lst.append(idx1)
                        idx_path.append(idx1)
                        branch = False
                    elif branch and subidx1 == -1: # no branch found
                        success = False 
                        break

                    if idx in idx_path or fd.ispit(flwdir_lr_flat[idx_ds1]): #  or checkd_lr_flat[idx_ds1] == 1
                        continue
                    elif subidx == outlet_lr_flat[idx]: # if original outlet -> try moving it to a branch
                        upa0 = 0
                        subidx1 = -1
                        branch = True
                    idx1 = idx
                if branch:
                    for subidx_us in fd.us_indices(subidx, flwdir_flat, shape):
                        idx_us = subidx_2_idx(subidx_us, shape[1], scale_ratio)
                        if idx_us == idx1 and subidx_us not in path and uparea_flat[subidx_us] > upa0:
                            subidx1 = subidx_us 
                            upa0 = uparea_flat[subidx_us]

        if success:
            for idx_ds in idx_path:
                # check if all upstream cells connect
                for idx_us in fd.us_indices(idx_ds, flwdir_lr_flat, shape_lr):
                    if idx_us in idx0_lst: continue
                    # outlet original upstream cell
                    subidx_ds = outlet_lr_flat[idx_us]
                    if idx_us in idx1_lst:
                        subidx_ds = out1_lst[np.where(np.array(idx1_lst)==idx_us)[0][0]]
                    subidx0 = -1
                    idx1 = idx_us
                    # move to next downstream outlet
                    while subidx0 != subidx_ds:
                        subidx = subidx_ds
                        subidx_ds = fd.ds_index(subidx, flwdir_flat, shape)
                        if subidx_ds == -1 or subidx_ds == subidx:
                            break
                        # lookup cells outlet
                        idx = subidx_2_idx(subidx_ds, shape[1], scale_ratio)
                        if idx != idx1:
                            idx1 = idx
                            subidx0 = outlet_lr_flat[idx1]
                            if idx1 in idx1_lst:
                                subidx0 = out1_lst[np.where(np.array(idx1_lst)==idx1)[0][0]]
                    # check if still connects if
                    dd1 = fd.idx_to_dd(idx_us, idx1, shape_lr)
                    if dd1 == fd._nodata or fd.ispit(dd1):           
                        success = False
                        if idx_ds not in bottleneck:
                            bottleneck.append(idx_ds)
                        break
                    elif idx1 != idx_ds:
                        idx0_lst.append(idx_us)
                        dd0_lst.append(dd1)

        if success: 
            idx1 = idx0
            loop = False
            check_loop = []
            while loop == False:
                if idx1 in idx0_lst:
                    dd1 = dd0_lst[np.where(np.array(idx0_lst)==idx1)[0][0]]
                    idx_ds1 = fd.ds_index(idx1, flwdir_lr_flat, shape_lr, dd=dd1)
                else:
                    idx_ds1 = fd.ds_index(idx1, flwdir_lr_flat, shape_lr)
                if idx_ds1 == idx1 or idx_ds1 == -1: break
                loop = idx_ds1 in check_loop
                check_loop.append(idx_ds1)
                idx1 = idx_ds1
            success = loop == False
            if success: break

        if len(bottleneck) > nb:
            nb = len(bottleneck)
        elif n != 0:
            n += 1
            if n >= 3:
                n = 0
        else:
            break

    return success, np.array(idx0_lst), np.array(dd0_lst), np.array(idx1_lst), np.array(out1_lst), np.array(bottleneck)



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
def d8_scaling(scale_ratio, flwdir_flat, uparea_flat, shape, upa_min=0.5, extended=True):
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
    idxs_pit = []
    subidxs_pit = []
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
        subidx0 = outlet_lr_flat[idx0]
        dd, subidx_out, idx_ds = _dd(idx0, subidx0, flwdir_flat, effare_flat, shape, shape_lr, scale_ratio, extended=extended)
        if dd != _nodata and effare_flat[subidx_out] <= np.uint(1): # subgrid outlets might not be connected
            # check if not connected
            ea = np.uint8(1)
            subidx = subidx0
            while ea != np.uint8(3):
                subidx_ds = fd.ds_index(subidx, flwdir_flat, shape)
                if subidx_ds == -1 or subidx_ds == subidx: break
                ea = effare_flat[subidx_ds]
                subidx = subidx_ds
            idx_ds = subidx_2_idx(subidx, shape[1], scale_ratio)
            if  fd.idx_to_dd(idx0, idx_ds, shape_lr) != dd: # not connected
                idx_check.append(idx0)
                upa_check.append(uparea_flat[subidx0])
                changd_lr_flat[idx0] = np.uint(1)
        if subidx_out != -1 and fd.ispit(dd) and fd.ispit(flwdir_flat[subidx0]) == False: # keep list of cells with outlet == pits outside own cell
            assert fd.ispit(flwdir_flat[subidx_out])
            idxs_pit.append(idx0)
            subidxs_pit.append(subidx_out)
        flwdir_lr_flat[idx0] = dd

    # second iteration to check misconnected
    if extended:
        loop = False
        seq = np.argsort(np.array(upa_check)) # from small to large
        for i in seq:
            upa0 = upa_check[i]
            # if upa0 < 1: continue
            idx0 = idx_check[i]
            success, idxs_dd, dds, idxs_out, subidxs_out, _ = _ddext(
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

    # fix double outlets -> assign to largest upstream cell
    if len(idxs_pit) == 1:
         outlet_lr_flat[idxs_pit[0]] = subidxs_pit[0]
    elif len(idxs_pit) > 1:
        idxs = np.array(idxs_pit)
        subidxs = np.array(subidxs_pit)
        isort = np.argsort(subidxs) # sort pits to find doubles
        subidxs_sort =  subidxs[isort]
        idxs_sort = idxs[isort]
        # print(idxs_sort)
        # print(subidxs_sort)
        j = 0
        for i, subidx in enumerate(subidxs_sort[1:]):
            if i >= j and subidxs_sort[i] == subidx: # if new double pit
                idxs_lst = []
                j = i
                while subidxs_sort[j] == subidx: # find all subsequent double pit
                    idxs_lst.append(idxs_sort[j])
                    j += 1
                idxs0 = np.array(idxs_lst)
                # print(idxs0)
                subidxs = outlet_lr_flat[idxs0] # get original outlet
                imax = np.argmax(uparea_flat[subidxs]) # find outlet with largest uparea
                idx_out0 = subidx_2_idx(subidx, shape[1], scale_ratio)
                for k in range(idxs0.size):
                    idx = idxs0[k]
                    if k == imax: # set pit to cell with largest upstream area
                        outlet_lr_flat[idx] = subidx #NOTE this pit is located outside cell idx
                    else: # set flowdir in other cells to drain to pit cell
                        dd = fd.idx_to_dd(idx, idx_out0, shape_lr)
                        assert dd != fd._nodata # make sure cells connect in d8
                        flwdir_lr_flat[idx] = dd
            elif i >= j:
                outlet_lr_flat[idxs_sort[i]] = subidxs_sort[i]
                if i+2 == idxs.size: # also set outlet index for final idx
                    outlet_lr_flat[idxs_sort[i+1]] = subidxs_sort[i+1]

            

    # # second dd iteration to change outlets if near tributary
    if extended:
        for idx0 in range(flwdir_lr_flat.size):
            if flwdir_lr_flat[idx0] == _nodata: continue
            idx_us_main, dd_new, subidx_out_new = _ddplus(
                idx0, flwdir_lr_flat, outlet_lr_flat, flwdir_flat, uparea_flat, effare_flat, 
                shape, shape_lr, scale_ratio, upa_min=4
            )
            if dd_new != _nodata:
                flwdir_lr_flat[idx_us_main] = dd_new
                outlet_lr_flat[idx0] = subidx_out_new
                if changd_lr_flat[idx0] == 0: 
                    changd_lr_flat[idx0] = np.uint(3)
    
    # assert outlet_lr_flat.size == np.unique(outlet_lr_flat).size
    # assert flwdir_check(flwdir_lr_flat, shape_lr)[1] == False # no loops
    return flwdir_lr_flat.reshape(shape_lr), outlet_lr_flat.reshape(shape_lr), changd_lr_flat.reshape(shape_lr)
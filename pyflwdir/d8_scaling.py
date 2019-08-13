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
    # calc lowres index of subgrid cell
    ncol = int(sub_ncol/scale_ratio)
    r = (subidx // sub_ncol) // scale_ratio
    c = (subidx %  sub_ncol) // scale_ratio
    return np.uint32(r * ncol + c)

@njit
def tilesubidx_2_subidx(tilesubidx, tileidx, sub_ncol, scale_ratio):
    ncol = int(sub_ncol/scale_ratio)
    r = tileidx // ncol * scale_ratio + tilesubidx // scale_ratio
    c = tileidx %  ncol * scale_ratio + tilesubidx %  scale_ratio
    return np.uint32(r * sub_ncol + c)

@njit
def _rep_cell(idx0, flwdir_flat, uparea_flat, shape, scale_ratio):
    R = scale_ratio/2.
    rr = np.abs(np.arange(-R+0.5, R, 1.))
    sub_ncol = shape[1]
    upa0 = np.float32(0) #
    subidx0 = np.uint32(-1)
    idx = np.uint32(-1)
    repcells = []
    for i in rr:
        for j in rr:
            idx += 1
            subidx = tilesubidx_2_subidx(idx, idx0, sub_ncol, scale_ratio)
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
        subidx0 = tilesubidx_2_subidx(idx, idx0, sub_ncol, scale_ratio)
    return subidx0, np.array(repcells)

@njit
def _outlet(idx0, subidx, flwdir_flat, shape, scale_ratio):
    sub_ncol = shape[1]
    subidx_out = np.uint32(-1)
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
    sub_ncol = shape[1]
    ncol = int(sub_ncol/scale_ratio)
    dd = _pits[0]
    i = 0
    while True:
        subidx_ds = fd.ds_index(subidx, flwdir_flat, shape) # move downstream
        if subidx_ds == np.uint32(-1): # outside domain
            break 
        idx_ds = subidx_2_idx(subidx_ds, sub_ncol, scale_ratio)
        # at pit/mouth
        if subidx_ds == subidx:
            dd = flwdir_flat[subidx]
            break
        # center lowres cell
        elif idx_ds == idx0:
            pass
        # outside 3x3 lowres neighbors
        elif abs(idx_ds-idx0) > 1 and abs(idx_ds-idx0-ncol) > 1 and abs(idx_ds-idx0+ncol) > 1: 
            break
        # in neighboring lowres cell
        else:  
            # check if in subgrid hits outlet / eff area
            flag = effare_flat[subidx_ds] # 1 eff area; 2 outlet
            if  flag >= np.uint8(1):
                dd = fd.idx_to_dd(idx0, idx_ds, shape_lr)
                assert dd != _nodata
                if not extended or flag >= np.uint8(2):
                    break
                i += 1
        # next iteration
        subidx = subidx_ds
    return dd

@njit
def _ddplus(idx0, flwdir_lr_flat, outlet_lr_flat, flwdir_flat, uparea_flat, effare_flat, 
            shape, shape_lr, scale_ratio, upa_min=0.5):
    idx_us_main = idx0
    dd_new = _nodata
    subidx_out = outlet_lr_flat[idx0]
    subidx_out_new = subidx_out
    # check if valid upstream & downstream cells
    idxs_us = fd.us_indices(idx0, flwdir_lr_flat, shape_lr)
    idx_ds = fd.ds_index(idx0, flwdir_lr_flat, shape_lr)
    if idxs_us.size == 0 or np.any(idxs_us<0) or idx_ds == idx0 or idx_ds < 0:
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
        if subidx_ds == subidx_us or subidx_ds < 0:
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
                invalid = subidx_ds == subidx_us or subidx_ds < 0 or effare_flat[subidx_ds] == np.uint8(3)
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
def subbasin_outlets_grid(scale_ratio, flwdir, uparea):
    sub_nrow, sub_ncol = uparea.shape
    lr_nrow, lr_ncol = int(sub_nrow/scale_ratio), int(sub_ncol/scale_ratio)
    shape_lr = lr_nrow, lr_ncol
    size_lr = lr_nrow * lr_ncol
    # subgrid input
    shape = flwdir.shape
    flwdir_flat = flwdir.reshape(-1)
    uparea_flat = uparea.reshape(-1)
    # output cells
    outlet_lr_flat = np.ones(size_lr, dtype=np.int64)*-1

    # get representative cells (largest uparea in effective area) on highres maps
    # trace to lowres cell boundary to find outlet cells
    for idx0 in range(outlet_lr_flat.size):
        subidx, _ = _rep_cell(idx0, flwdir_flat, uparea_flat, shape, scale_ratio)
        subidx_out, _ = _outlet(idx0, subidx, flwdir_flat, shape, scale_ratio)
        outlet_lr_flat[idx0] = subidx_out   # subgrid index of outlet point lowres grid cell

    return outlet_lr_flat.reshape(shape_lr)


@njit
def d8_scaling(scale_ratio, flwdir, uparea, upa_min=0.5, extended=True):
    sub_nrow, sub_ncol = uparea.shape
    lr_nrow, lr_ncol = int(sub_nrow/scale_ratio), int(sub_ncol/scale_ratio)
    shape_lr = lr_nrow, lr_ncol
    size_lr = lr_nrow * lr_ncol
    # subgrid input
    shape = flwdir.shape
    flwdir_flat = flwdir.reshape(-1)
    uparea_flat = uparea.reshape(-1)
    # output cells
    outlet_lr_flat = np.ones(size_lr, dtype=np.uint32)*-1
    repcel_lr_flat = np.ones(size_lr, dtype=np.uint32)*-1
    flwdir_lr_flat = np.zeros(size_lr, dtype=flwdir.dtype)
    effare_flat = np.zeros(flwdir.size, dtype=np.uint8)    

    # get effective area and representative cells highres maps
    # and index of rep cell on lowres map
    for idx0 in range(repcel_lr_flat.size):
        subidx_rep, subidxs_ea = _rep_cell(idx0, flwdir_flat, uparea_flat, shape, scale_ratio)
        repcel_lr_flat[idx0] = subidx_rep
        effare_flat[subidxs_ea] = np.uint(1) # save effective area

    # find outlet cells and save to highres and lowres grids
    for idx0 in range(outlet_lr_flat.size):
        subidx = repcel_lr_flat[idx0]
        subidx_out, subidxs_rc = _outlet(idx0, subidx, flwdir_flat, shape, scale_ratio)
        outlet_lr_flat[idx0] = subidx_out   # subgrid index of outlet point lowres grid cell
        if extended:
            effare_flat[subidxs_rc] = np.uint8(2) # 2 for representative stream cells (between representative cell and outlet)
            effare_flat[subidx_out] = np.uint8(3)

    for idx0 in range(flwdir_lr_flat.size):
        subidx = outlet_lr_flat[idx0]
        dd = _dd(idx0, subidx, flwdir_flat, effare_flat, shape, shape_lr, scale_ratio, extended=extended)
        flwdir_lr_flat[idx0] = dd

    # TODO correct for multiple outlets per basin. In case the basin outlet is not set a cell outlet, 
    # i.e. when another larger stream passes through the same cell, multiple upstream cells may connect
    # to the basin outlet cell. The cell with the smallest uparea at the cell outlet should be rerouted 
    # through the other cell. 

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
    assert outlet_lr_flat.size == np.unique(outlet_lr_flat).size
    assert flwdir_check(flwdir_lr_flat, shape_lr)[1] == False # no loops
    return flwdir_lr_flat.reshape(shape_lr), outlet_lr_flat.reshape(shape_lr)
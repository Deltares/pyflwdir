# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

from numba import njit
import numpy as np

# set d8 format
from d8_func import d8
_pits = d8._pits
_nodata = d8._nodata

@njit
def subidx_2_idx(subidx, sub_ncol, scale_ratio):
    # calc lowres index of subgrid cell
    ncol = int(sub_ncol/scale_ratio)
    r = (subidx // sub_ncol) // scale_ratio
    c = (subidx %  sub_ncol) // scale_ratio
    return r * ncol + c 

@njit
def tilesubidx_2_subidx(tilesubidx, tileidx, sub_ncol, scale_ratio):
    ncol = int(sub_ncol/scale_ratio)
    r = tileidx // ncol * scale_ratio + tilesubidx // scale_ratio
    c = tileidx %  ncol * scale_ratio + tilesubidx %  scale_ratio
    return r * sub_ncol + c

@njit
def _rep_cell(idx0, flwdir_flat, uparea_flat, shape, scale_ratio):
    R = scale_ratio/2.
    rr = np.abs(np.arange(-R+0.5, R, 1.))
    sub_ncol = shape[1]
    upa0 = 0 #
    subidx0 = -1
    idx = -1
    repcells = []
    for i in rr:
        for j in rr:
            idx += 1
            subidx = tilesubidx_2_subidx(idx, idx0, sub_ncol, scale_ratio)
            ispit = d8.ispit(flwdir_flat[subidx])
            eff_area = (i**0.5 + j**0.5 <= R**0.5) or (i <= 0.5) or (j <= 0.5) # describes effective area
            if ispit or eff_area: 
                upa = uparea_flat[subidx]
                if eff_area:  
                    repcells.append(subidx)
                if upa > upa0:
                    upa0 = upa
                    subidx0 = subidx
    
    if subidx0 != -1 and d8.ispit(flwdir_flat[subidx0]) and np.all(np.array(repcells)!=subidx0):
        repcells.append(subidx0)
    elif subidx0 == -1:
        # select center cell if no rep cell found
        idx = int(round((scale_ratio**2)/2.))
        subidx0 = tilesubidx_2_subidx(idx, idx0, sub_ncol, scale_ratio)
    return subidx0, np.array(repcells)

@njit
def _outlet(idx0, subidx, flwdir_flat, shape, scale_ratio):
    sub_ncol = shape[1]
    subidx_out = -1
    idx = idx0
    streamcells = [subidx]
    while True:
        subidx_ds = d8.ds_d8(subidx, flwdir_flat, shape) # move downstream
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
        subidx_ds = d8.ds_d8(subidx, flwdir_flat, shape) # move downstream
        if subidx_ds == -1: # outside domain
            break 
        idx_ds = subidx_2_idx(subidx_ds, sub_ncol, scale_ratio)
        # at pit/mouth
        if subidx_ds == subidx: 
            # dd = d8.idx_to_d8(idx0, idx_ds, shape_lr)
            # if dd == _pits[0]:
            #     dd = flwdir_flat[subidx]
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
                dd = d8.idx_to_d8(idx0, idx_ds, shape_lr)
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
    idxs_us = d8.us_d8(idx0, flwdir_lr_flat, shape_lr)
    idx_ds = d8.ds_d8(idx0, flwdir_lr_flat, shape_lr)
    if idxs_us.size == 0 or np.any(idxs_us<0) or idx_ds == idx0 or idx_ds < 0:
        return idx0, _nodata, subidx_out
    # check if main us connects to next ds in d8
    subidxs_out_us = outlet_lr_flat[idxs_us]
    maxi = np.argmax(uparea_flat[subidxs_out_us])
    idx_us_main = idxs_us[maxi]
    dd_new = d8.idx_to_d8(idx_us_main, idx_ds, shape_lr)
    assert not d8.ispit(dd_new)
    if dd_new == _nodata: # -1 if it does not connect in d8
        return idx0, _nodata, subidx_out
    # trace main stream downstream
    subidx_us = subidxs_out_us[maxi]
    mainstream = [subidx_us]
    while True:
        subidx_ds = d8.ds_d8(subidx_us, flwdir_flat, shape)
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
        subidxs_us = d8.us_d8(subidx_ds, flwdir_flat, shape)
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
                subidx_ds = d8.ds_d8(subidx_us, flwdir_flat, shape)
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
def d8_scaling(scale_ratio, flwdir, uparea, upa_min=0.5, extended=True):
    sub_nrow, sub_ncol = uparea.shape
    lr_nrow, lr_ncol = int(sub_nrow/scale_ratio), int(sub_ncol/scale_ratio)
    shape_lr = lr_nrow, lr_ncol
    size_lr = lr_nrow * lr_ncol
    # subgrid input
    shape = flwdir.shape
    flwdir_flat = flwdir.flatten()
    uparea_flat = uparea.flatten()
    # output cells
    outlet_lr_flat = np.ones(size_lr, dtype=np.int64)*-1
    repcel_lr_flat = np.ones(size_lr, dtype=np.int64)*-1
    flwdir_lr_flat = np.zeros(shape_lr, dtype=flwdir.dtype).flatten()
    effare_flat = np.zeros(shape, dtype=np.uint8).flatten()    

    # get effective area and representative cells highres maps
    # and index of rep cell on lowres map
    for idx0 in range(repcel_lr_flat.size):
        subidx_rep, subidxs_ea = _rep_cell(idx0, flwdir_flat, uparea_flat, shape, scale_ratio)
        repcel_lr_flat[idx0] = subidx_rep
        effare_flat[subidxs_ea] = np.uint(1) # save effective area
    assert repcel_lr_flat.size == np.unique(repcel_lr_flat).size

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
    assert check_flwdir(flwdir_lr_flat, shape_lr).size==0

    return flwdir_lr_flat.reshape(shape_lr), outlet_lr_flat.reshape(shape_lr), repcel_lr_flat.reshape(shape_lr)

# TODO: check that all cells drain to an outlet.
# On a tile level (with buffer) this could be done by checking 
# that cells reach either the tile bounds or an outlet
@njit
def check_flwdir(flwdir_lr_flat, shape_lr):
    nrow, ncol = shape_lr
    idx_wrong = []
    for idx0 in range(flwdir_lr_flat.size):
        # skip edges 
        r0 = idx0 // ncol
        c0 = idx0 %  ncol
        if r0 == 0 or r0 == nrow-1 or c0 == 0 or c0 == ncol-1:
            continue
        idx_ds = d8.ds_d8(idx0, flwdir_lr_flat, shape_lr)
        if idx_ds == -1 or idx_ds == idx0:
            continue
        idx_ds2 = d8.ds_d8(idx_ds, flwdir_lr_flat, shape_lr)
        if idx0 == idx_ds2:
            idx_wrong.append(idx0)
    return np.array(idx_wrong, dtype=np.int64)

def get_upareas(flwdir_lr, outlet_lr, uparea, flwdir, scale_ratio):
    from d8_flux import accuflux
    from d8_scaling_subgrid import group_count, subgrid_indices
    # get highres upgrid at outlet
    mask_lr = flwdir_lr!=_nodata
    subidx_out = outlet_lr[mask_lr]
    upgrid_out = np.ones(mask_lr.shape, dtype=np.float32)*-9999
    upgrid_out[mask_lr] = uparea.flat[subidx_out].astype(np.float32)
    # caclute upgrid
    idxs_ds = np.where(np.logical_or(flwdir_lr.flatten()==0, flwdir_lr.flatten()==255))[0]
    print(idxs_ds)
    if not idxs_ds.size == 1:
        import pdb; pdb.set_trace()
    outlet_lr = outlet_lr.copy()
    outlet_lr[~mask_lr] = -9999
    groups, indices, _ = subgrid_indices(outlet_lr, flwdir, uparea, upa_min=0.5)
    catare = np.ones(mask_lr.shape, dtype=np.float32)
    catare[mask_lr] = group_count(groups, indices).astype(np.float32)
    upgrid_lr = accuflux(flwdir=flwdir_lr, material=catare, idxs_ds=idxs_ds).astype(np.float32)
    upgrid_lr[~mask_lr] = -9999
    return upgrid_out, upgrid_lr, mask_lr

if __name__ == "__main__":
    import xarray as xr
    import pandas as pd
    from os.path import join, isfile
    import numpy as np
    import matplotlib.pyplot as plt
    # local libraries
    from tif_io import open_vrt_dataset
    from d8_scaling_performance import test_scaling

    test = True
    root = r'/media/data/hydro_merit_1.0'
    # root = r'd:/work/flwdir_scaling'
    merit_dir = join(root, r'03sec')
    fn_outlets = join(merit_dir, f'all_outlets_upa100_pfaf.csv')
    df = pd.read_csv(fn_outlets, index_col=0)
    res = 1/1200.
    buf = 0.25
    
    # select basin 
    # WARNING: basin(s) 2941, 10599, 13074, 13666, 14549, 15276 have multiple outlets
    idx = 102 #102 #931 #74: rhine
    bas = df.loc[idx, ]
    w,s,e,n = bas[['xmin', 'ymin', 'xmax', 'ymax']].values
    w,s,e,n = w-buf, max(s-buf,-60), e+buf, min(n+buf,85.)
    w,s,e,n = np.floor(w*4)/4.,np.floor(s*4)/4., np.ceil(e*4)/4., np.ceil(n*4)/4.
    print(f'{idx:d}: ({w:.3f}, {s:.3f}, {e:.3f}, {n:.3f})')

    test_fn = join(merit_dir, f'test_sel_idx{idx:d}.nc')
    if test and isfile(test_fn):
        ds = xr.open_dataset(join(merit_dir, f'test_sel_idx{idx:d}.nc')).load()
    else:
        ds = open_vrt_dataset(merit_dir, ['upa', 'dir', 'bas', 'upg'])
    ds_sel = ds.sel(lon=slice(w, e), lat=slice(n, s))
    mask = ds_sel['bas'].load().values != idx
    flwdir = ds_sel['dir'].load().values
    flwdir[mask] = np.uint8(247)
    uparea = ds_sel['upa'].load().values
    upgrid = ds_sel['upg'].load().values
    height, width = uparea.shape
    print(f'{idx:d}: ({height:d}x{width:d})')
    if test and not isfile(test_fn):
        ds_sel.to_netcdf(test_fn)

    flwdir_flat = flwdir.flatten()
    idxs_ds = np.where(np.logical_or(flwdir_flat==0, flwdir_flat==255))[0]
    print(idxs_ds)
    assert idxs_ds.size == 1
    for scale_ratio in [20]: #[10, 20, 100, 300]:
        out_res = scale_ratio*res
        if 3600/(1/out_res) < 60:
            resname = '{:02.0f}sec'.format(3600/(1/out_res))
        else:
            resname = '{:02.0f}min'.format(60/(1/out_res))
        print(f'out res = {resname:s} ({height/scale_ratio:.0f}, {width/scale_ratio:.0f})')

        # flwdir_lr, outlet_lr, _ = d8_scaling(scale_ratio, flwdir, uparea, extended=False)
        # upgrid_out, upgrid_lr, mask_lr = get_upareas(flwdir_lr, outlet_lr, upgrid, flwdir, scale_ratio)
        # nse = test_scaling(upgrid_lr[mask_lr], upgrid_out[mask_lr])
        # print(f'EEA original nse: {nse:.4f}')

        flwdir_lr_ext, outlet_lr_ext, _ = d8_scaling(scale_ratio, flwdir, uparea, extended=True)
        upgrid_out_ext, upgrid_lr_ext, mask_lr_ext = get_upareas(flwdir_lr_ext, outlet_lr_ext, upgrid, flwdir, scale_ratio)
        nse_ext = test_scaling(upgrid_lr_ext[mask_lr_ext], upgrid_out_ext[mask_lr_ext])
        print(f'EEA ext nse: {nse_ext:.4f}')

        # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), sharey=True)
        # ax1.set_aspect('equal')
        # ax2.set_aspect('equal')
        # ax1.set_yscale('log')
        # ax1.set_xscale('log')
        # ax2.set_yscale('log')
        # ax2.set_xscale('log')
        # ax1.scatter(upgrid_lr, upgrid_out, color='k', s=6, marker='o', label=f'original; NSE: {nse:.4f}')
        # ax2.scatter(upgrid_lr_ext, upgrid_out_ext, color='r', s=6, marker='*', label=f'extended; NSE: {nse_ext:.4f}')
        # ax2.legend(loc='lower right')
        # ax1.legend(loc='lower right')
        # ax1.set_xlabel('upgrid at outlet [no. cells]')
        # ax2.set_xlabel('upgrid at outlet [no. cells]')
        # ax1.set_ylabel('scaled upgrid [no. cells]')
        # ax1.set_title(f'EEA original: res: {resname}')
        # ax2.set_title(f'EEA extended: res: {resname}')
        # fn = join(merit_dir, '../plots', f'basin{idx:03d}_{resname}.png')
        # plt.savefig(fn, dpi=255, bbox_axis='tight')

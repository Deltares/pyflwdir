# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

"""Tests for the pyflwdir module.
"""
import pytest
from affine import Affine
import time

import numpy as np

from pyflwdir import core, core_conversion, upscale, FlwdirRaster, gis_utils
from pyflwdir.basin_utils import basin_area
_mv = core._mv 

def _covariance(x, y):
    return np.nanmean((x - np.nanmean(x, axis=-1, keepdims=True))
            * (y - np.nanmean(y, axis=-1, keepdims=True)), axis=-1)
def _pearson_correlation(x, y):
    return _covariance(x, y) / (np.nanstd(x, axis=-1) * np.nanstd(y, axis=-1))
def _rsquared(x, y):
    return _pearson_correlation(x, y) ** 2
def _nse(sim, obs, axis=-1):
    """nash-sutcliffe efficiency"""
    obs_mean = np.nanmean(obs, axis=axis)
    a = np.nansum((sim-obs)**2, axis=axis)
    b = np.nansum((obs-obs_mean[..., None])**2, axis=axis)
    return 1 - a/b

@pytest.fixture
def d8_data():
    return np.fromfile(r'./tests/data/ireland_d8.bin', dtype=np.uint8).reshape((4920, 6120))
@pytest.fixture
def d8(d8_data):
    return FlwdirRaster(d8_data, ftype = 'd8', check_ftype = False)

def test_eam(d8):
    res = 1/1200.
    cellsize = 10
    affine = Affine(res, 0.0, -10.5, 0.0, -res, 55.5)
    uparea = d8.upstream_area(affine=affine, latlon=True).ravel()
    nextidx, subidxs_out, shape = upscale.eam_upscale(d8._idxs_ds, d8._idxs_valid, uparea, d8.shape, cellsize)
    flwdir = core_conversion.nextidx_to_d8(nextidx, shape)
    d8_lr = FlwdirRaster(flwdir, ftype = 'd8', check_ftype = True)
    assert np.unique(nextidx[d8_lr.pits]).size == nextidx[d8_lr.pits].size
    assert d8_lr.isvalid
    # check quality
    valid = nextidx != _mv
    subbasins = d8.subbasins(subidxs_out[valid])
    subare = np.ones(shape)*-9999.
    subare.flat[valid] = basin_area(subbasins, affine=affine)
    uparea_lr = d8_lr.accuflux(subare).ravel()[valid]
    uparea_out = uparea[subidxs_out[valid]]
    nse = _nse(uparea_lr, uparea_out) 
    upadiff = np.abs(uparea_lr - uparea_out)
    relbias10 = np.sum(upadiff/uparea_out>0.1)/uparea_out.size*100
    assert np.round(nse, 4) == 0.9937, f'NSE: {nse:.4f} (0.9937)'
    assert np.round(relbias10, 4) == 12.3994, f'%cells with error rel. bias > 10%: {upadiff.max():.4f} (12.3994)'
    assert np.round(upadiff.max()/1e6,4) == 4868.9605, f'max abs. diff [km2]: {upadiff.max()/1e6:.4f} (4868.9605)'
    assert np.round(upadiff.mean()/1e6,4) == 1.8828, f'mean abs. diff [km2]: {upadiff.mean()/1e6:.4f} (1.8828)'
    
    # # write fils for visual check
    # import rasterio
    # xs, ys = np.ones(nextidx.size)*np.nan, np.ones(nextidx.size)*np.nan
    # xs[valid], ys[valid] = gis_utils.idxs_to_coords(subidxs_out[valid], affine, d8.shape)
    # d8_lr.vector(xs=xs, ys=ys).to_file(f'./tests/data/ireland{cellsize}_eam.gpkg', layer='rivers', driver="GPKG")
    
    # data = d8.subbasins(subidxs_out[valid])
    # name = 'ucat'
    # nodata = 0
    # data = d8.stream_order().astype(np.int32)
    # name = 'ord'
    # nodata = -1
    # prof = dict(
    #     driver = 'GTiff',
    #     transform = Affine(res, 0.0, -10.5, 0.0, -res, 55.5), 
    #     height = data.shape[0], 
    #     width = data.shape[1],
    #     count = 1,
    #     dtype = data.dtype,
    #     nodata = nodata
    #     )
    # with rasterio.open(f'./tests/data/ireland{cellsize}_{name}.tif', 'w', **prof) as src:
    #     src.write(data, 1)
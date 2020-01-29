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
    return np.nanmean((x - np.nanmean(x, axis=-1, keepdims=True)) *
                      (y - np.nanmean(y, axis=-1, keepdims=True)),
                      axis=-1)


def _pearson_correlation(x, y):
    return _covariance(x, y) / (np.nanstd(x, axis=-1) * np.nanstd(y, axis=-1))


def _rsquared(x, y):
    return _pearson_correlation(x, y)**2


def _nse(sim, obs, axis=-1):
    """nash-sutcliffe efficiency"""
    obs_mean = np.nanmean(obs, axis=axis)
    a = np.nansum((sim - obs)**2, axis=axis)
    b = np.nansum((obs - obs_mean[..., None])**2, axis=axis)
    return 1 - a / b


@pytest.fixture
def d8_data():
    return np.fromfile(r'./tests/data/ireland_d8.bin', dtype=np.uint8).reshape(
        (4920, 6120))


@pytest.fixture
def d8(d8_data):
    return FlwdirRaster(d8_data, ftype='d8', check_ftype=False)


scores = {  #NSE    relbias10 upadiffmax upadiffavg
    'eam': [0.9191, 4.0601, 22491.2400, 7.5579],
    'eeam': [0.9988, 0.3510, 5550.5100, 0.0341],  # unit 100 cells 
}


def test_upscale_scores(d8):
    res = 1 / 1200.
    cellsize = 10
    affine = Affine(res, 0.0, -10.5, 0.0, -res, 55.5)
    # uparea = d8.upstream_area(affine=affine, latlon=True).ravel()
    uparea = d8.upstream_area(latlon=False).ravel()
    uparea = np.where(uparea != -9999., uparea / 1e2, -9999.)
    basins = d8.basins().ravel()
    for method in ['eeam', 'eam'][:1]:
        ms = scores[method]
        fupscale = getattr(upscale, method)
        nextidx, subidxs_out, _ = fupscale(d8._idxs_ds, d8._idxs_valid, uparea,
                                           d8.shape, cellsize)
        dir_lr = FlwdirRaster(nextidx, ftype='nextidx', check_ftype=True)
        assert dir_lr.isvalid
        assert np.unique(
            nextidx.flat[dir_lr.pits]).size == nextidx.flat[dir_lr.pits].size
        pitbas = basins[subidxs_out[dir_lr.pits]]
        assert np.unique(pitbas).size == pitbas.size
        # check quality
        valid = nextidx.ravel() != _mv
        subbasins = d8.subbasins(subidxs_out[valid])
        subare = np.ones(dir_lr.shape) * -9999.
        # subare.flat[valid] = basin_area(subbasins, affine=affine, latlon=True)
        subare.flat[valid] = basin_area(subbasins, latlon=False)
        uparea_lr = dir_lr.accuflux(subare, nodata=-9999.).ravel()
        uparea_lr = np.where(uparea_lr != -9999., uparea_lr / 1e2, -9999.)
        uparea_out = np.ones(dir_lr.shape).ravel() * -9999.
        uparea_out[valid] = uparea[subidxs_out[valid]]
        nse = _nse(uparea_lr[valid], uparea_out[valid])
        upadiff = np.where(uparea_lr != -9999, (uparea_out - uparea_lr), -9999)
        relbias10 = np.sum(
            upadiff[valid] / uparea_out[valid] > 0.1) / upadiff.size * 100
        s = [nse, relbias10, upadiff[valid].max(), upadiff[valid].mean()]

        # write files for visual check
        import rasterio
        xs, ys = np.ones(nextidx.size) * np.nan, np.ones(nextidx.size) * np.nan
        xs[valid], ys[valid] = gis_utils.idxs_to_coords(
            subidxs_out[valid], affine, d8.shape)
        dir_lr.vector(xs=xs, ys=ys).to_file(
            f'./tests/data/ireland{cellsize}_{method}.gpkg',
            layer='rivers',
            driver="GPKG")
        nodata = -9999.
        data = upadiff.reshape(dir_lr.shape)
        name = f'upadiff_{method}'
        transform = Affine(res * cellsize, 0.0, -10.5, 0.0, -res * cellsize,
                           55.5)
        # data = d8.subbasins(subidxs_out[valid])
        # name = 'ucat'
        # nodata = 0
        # data = d8.stream_order().astype(np.int32)
        # name = 'ord'
        # nodata = -1
        # transform = affine
        prof = dict(driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    transform=transform,
                    count=1,
                    dtype=data.dtype,
                    nodata=nodata)
        with rasterio.open(f'./tests/data/ireland{cellsize}_{name}.tif', 'w',
                           **prof) as src:
            src.write(data, 1)

        msg = f'NSE: {s[0]:.4f} ({ms[0]:.4f}); %cells rel. bias > 10%: {s[1]:.4f} ({ms[1]:.4f});' +\
        f'max abs. diff [km2]: {s[2]:.4f} ({ms[2]:.4f}); mean abs. diff [km2]: {s[3]:.4f} ({ms[3]:.4f})'
        assert np.all([np.round(s[i], 4) == ms[i] for i in range(len(s))]), msg


# def test_convf(d8):
#     import rasterio
#     res = 1/1200.
#     cellsize = 10
#     affine = Affine(res, 0.0, -10.5, 0.0, -res, 55.5)
#     celledge = upscale.map_celledge(d8._idxs_ds, d8._idxs_valid, d8.shape, cellsize).astype(np.uint8)
#     effare = upscale.map_effare(d8._idxs_ds, d8._idxs_valid, d8.shape, cellsize).astype(np.uint8)
#     nodata = 255
#     prof = dict(
#         driver = 'GTiff',
#         height = d8.shape[0],
#         width = d8.shape[1],
#         transform = affine,
#         count = 1,
#         dtype = effare.dtype,
#         nodata = nodata
#         )

#     with rasterio.open(f'./tests/data/ireland{cellsize}_effare.tif', 'w', **prof) as src:
#         src.write(effare, 1)
#     with rasterio.open(f'./tests/data/ireland{cellsize}_celledge.tif', 'w', **prof) as src:
#         src.write(celledge, 1)

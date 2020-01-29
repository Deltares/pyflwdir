# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Tests for the pyflwdir module.
"""
import pytest
from affine import Affine
import time
import numpy as np
# local 
from pyflwdir import (
    core, 
    core_conversion, 
    gis_utils,
    FlwdirRaster, 
    upscale,
    subgrid
    )
_mv = core._mv


@pytest.fixture
def d8_data():
    data = np.fromfile(r'./tests/data/d8.bin', dtype=np.uint8)
    return data.reshape((678, 776))


@pytest.fixture
def d8(d8_data):
    return FlwdirRaster(d8_data, ftype='d8', check_ftype=False)


# configure tests with different options 
# n is number of disconnected cells per method
tests = {
    'dmm': {'n': 1073}, 
    'eam': {'n': 424}, 
    'cosm': {'kwargs': {'iter2': False}, 'n': 144}, 
    'cosm2': {'method': 'cosm', 'n': 67}, 
}


def test_upscale(d8):
    cellsize = 10
    uparea = d8.upstream_area(latlon=False).ravel()
    basins = d8.basins().ravel()
    for name, mdict in tests.items():
        fupscale = getattr(upscale, mdict.get('method', name))
        kwargs = mdict.get('kwargs', {})
        nextidx, subidxs_out = fupscale(d8._idxs_ds, d8._idxs_valid, uparea,
                                           d8.shape, cellsize, **kwargs)
        d8_lr = FlwdirRaster(nextidx, ftype='nextidx', check_ftype=True)
        subidxs_out = subidxs_out.ravel()
        assert d8_lr.isvalid
        pit_idxs = nextidx.flat[d8_lr.pits]
        assert np.unique(pit_idxs).size == pit_idxs.size, name
        pitbas = basins[subidxs_out[d8_lr.pits]]
        assert np.unique(pitbas).size == pitbas.size, name
        # check number of disconnected cells for each method
        subidx_out_internal = d8._internal_idx(subidxs_out[d8_lr._idxs_valid])
        connect = subgrid.connected(d8_lr._idxs_ds, subidx_out_internal, 
                                    d8._idxs_ds)
        print(name, np.sum(connect == 0))
        assert np.sum(connect == 0) == mdict['n'], name


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

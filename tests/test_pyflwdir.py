# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

"""Tests for the pyflwdir module.
"""
import pytest

import numpy as np
import xarray as xr
import rasterio 

import pyflwdir
from pyflwdir import FlwdirRaster
from pyflwdir.core import fd
from pyflwdir.utils import flwdir_check

ds = xr.open_dataset(r'./tests/data/thur_test.nc')
raster = ds['flwdir'].values
nrow, ncol = raster.shape
res = 1/120.
w = np.round((ds['lon'].values.min()-res/2)/res)*res
n = np.round((ds['lat'].values.max()+res/2)/res)*res
idx0 = 864

prof = dict(
    crs={'init': 'epsg:4326'},
    width=ncol,
    height=nrow,
    count=1,
    transform = rasterio.transform.from_origin(w, n, res, res),
    driver='GTiff',
    dtype = np.int32,
    nodata = -9999
)

# def test_something():
#     assert True

# def test_with_error():
#     with pytest.raises(ValueError):
#         # Do something that raises a ValueError
#         raise(ValueError)

# # Fixture example
# @pytest.fixture
# def an_object():
#     return {}

# test object
def test_object():
    flwdir = FlwdirRaster(raster)
    assert isinstance(flwdir, FlwdirRaster)

def test_flwdir_repair():
    flwdir = FlwdirRaster(raster)
    assert flwdir.isvalid()
    lst, hasloops = flwdir_check(flwdir.data)
    assert len(lst) == 110 and not hasloops
    flwdir.repair() # repair edges
    lst, hasloops = flwdir_check(flwdir.data)
    assert len(lst) == 0 and not hasloops
    # create loop
    idx = 450 
    idx_us = fd.us_indices(idx, flwdir.data_flat, flwdir.shape)[0]
    flwdir[idx] = fd.idx_to_dd(idx, idx_us, flwdir.shape)
    # import pdb; pdb.set_trace() 
    lst, hasloops = flwdir_check(flwdir.data)
    assert len(lst) == 1 and lst[0] == idx and hasloops
    flwdir.repair() # repair loop
    lst, hasloops = flwdir_check(flwdir.data)
    assert len(lst) == 0 and not hasloops

def test_setup_network():
    flwdir = FlwdirRaster(raster)
    flwdir.repair()
    flwdir.setup_network()
    tot_n = np.sum([np.sum(n!=-1) for n in flwdir.rnodes_up]) + flwdir.rnodes[-1].size
    assert tot_n == flwdir.size

def test_basin_bounds():
    flwdir = FlwdirRaster(raster)
    flwdir.repair() # after repair the total bbox should be equal to flwdir.bbox
    bounds = flwdir.basin_bounds()
    xmin, ymin = bounds.min(axis=0)[[0,1]]
    xmax, ymax = bounds.max(axis=0)[[2,3]]
    assert np.all((xmin, ymin, xmax, ymax) == flwdir.bounds)

def test_basin_delination():
    # test single basin
    flwdir = FlwdirRaster(raster)
    flwdir.setup_network(idx0)
    basins = flwdir.basin_map()
    assert np.all(np.unique(basins[basins!=0])==1) # single basin index
    assert np.sum(basins) == 3045
    np.random.seed(0)
    idxs = np.where(basins.ravel())[0]
    idx = np.concatenate((
        idxs[np.random.randint(0, np.sum(basins), 18)],
        np.array([idx0])
        )
    )
    basins2 = flwdir.basin_map(idx=idx, values=idx)
    assert np.sum(basins2!=0) == 3045
    assert np.unique(basins2).size == 20 # 19 subbasins + background zero

def test_stream_order():
    # 
    flwdir = FlwdirRaster(raster)
    flwdir.setup_network(idx0)
    stream_order = flwdir.stream_order()
    assert np.unique(stream_order).size == 7
    assert np.sum(stream_order>0) == 3045
    assert np.sum(stream_order==6) == 88
    # with rasterio.open(r'../tests/data/stream_order.tif', 'w', **prof) as dst:
    #     dst.write(stream_order.astype(np.int32), 1)

def test_uparea():
    # test as if metres with identity transform
    flwdir = FlwdirRaster(raster, latlon=False)
    flwdir.setup_network(idx0)
    upa = flwdir.upstream_area()
    tot_n = np.sum([np.sum(n!=-1) for n in flwdir.rnodes_up]) + flwdir.rnodes[-1].size
    assert np.round(upa.max()*1e6,2) == tot_n == 3045
    # test in latlon with identity transform
    flwdir = FlwdirRaster(raster, latlon=True)
    flwdir.setup_network(idx0)
    upa = flwdir.upstream_area()
    assert np.round(upa.max(),8) == 31610442.71200391


# if __name__ == "__main__":
    # test_flwdir_repair()
    # test_setup_network()
    # test_basin_bounds()
    # test_uparea()
    # test_basin_delination()
    # test_stream_order()
    # test_stream_order()
    # import pdb; pdb.set_trace()

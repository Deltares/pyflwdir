# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Tests for the pyflwdir module.
"""
import pytest
import numba
import time
from affine import Affine
rtsys = numba.runtime.rtsys

import numpy as np

import pyflwdir
from pyflwdir.core import _mv

# read and parse data
@pytest.fixture
def d8_data():
    data = np.fromfile(r'./tests/data/d8.bin', dtype=np.uint8)
    return data.reshape((678, 776))

@pytest.fixture
def ldd_data():
    data = np.fromfile(r'./tests/data/ldd.bin', dtype=np.uint8)
    return data.reshape((678, 776))

@pytest.fixture
def nextxy_data():
    data = np.fromfile(r'./tests/data/nextxy.bin', dtype=np.int32)
    return data.reshape((2, 678, 776))

@pytest.fixture
def nextidx_data():
    data = np.fromfile(r'./tests/data/nextidx.bin', dtype=np.uint32)
    return data.reshape((678, 776))

@pytest.fixture
def d8():
    flw = pyflwdir.load(r'./tests/data/flw.pkl')
    flw._idxs_us # initialize us array
    return flw

# test object
def test_object(d8_data, ldd_data, nextxy_data, nextidx_data):
    ftypes = {
        'd8': d8_data, 
        'ldd': ldd_data, 
        'nextxy': nextxy_data, 
        'nextidx': nextidx_data
    }
    for name, fdata in ftypes.items():
        fd = pyflwdir.from_array(fdata)
        assert fd.ftype == name
        assert fd.size == 678 * 776
        assert fd.shape == (678, 776)
        assert fd._idxs_us_ is None
        assert fd._tree_ is None
        npits = fd._idxs_pit.size
        nus = fd._idxs_us[fd._idxs_us != _mv].size
        nds = fd._idxs_ds.size
        assert nus + npits == nds
        assert fd.pits.size == 1
        assert fd.isvalid
    # make sure these raise errors
    with pytest.raises(ValueError):
        pyflwdir.from_array(ldd_data, ftype='d8', check_ftype=True)  # invalid type
        pyflwdir.from_array(nextxy_data, ftype='d8', check_ftype=True)  # invalid shape & type
        pyflwdir.from_array(np.arange(20, dtype=np.uint32), ftype='infer', check_ftype=False)  # unknown type
        pyflwdir.from_array(np.array([2]), ftype='d8', check_ftype=True)  # invalid data: too small


def test_toarray(d8, d8_data):
    assert np.all(d8.to_array() == d8_data)

def test_stream_order(d8):
    try:
        strord = d8.stream_order()
    except:
        pytest.fail('stream order failed')
    assert strord.min() == -1
    assert strord.max() == 9  # NOTE specific to data
    assert strord.dtype == np.int8
    assert np.all(strord.shape == d8.shape)


def test_upstream_area(d8):
    # TODO add test with latlon
    try:
        uparea = d8.upstream_area()
    except:
        pytest.fail('upstream area failed')
    assert uparea.min() == -9999
    assert uparea[uparea != -9999].min() == 1
    assert uparea.max() == d8.ncells  # NOTE specific to data with single basin
    assert uparea.dtype == np.float64
    assert np.all(uparea.shape == d8.shape)


def test_basins(d8):
    try:
        basins = d8.basins()
    except:
        pytest.fail('basins failed')
    assert basins.min() == 0
    assert basins.max() == 1  # NOTE specific to data with single basin
    assert basins.sum() == d8.ncells  # NOTE specific to data with single basin
    assert basins.dtype == np.uint32
    assert np.all(basins.shape == d8.shape)

def test_upscale(d8, nextidx_data):
    try:
        d8_lr, idxout = d8.upscale(10)
    except:
        pytest.fail('upscaling failed')
    with pytest.raises(ValueError):
        d8.upscale(10, method='unknown') # unknown method
        pyflwdir.from_array(nextidx, ftype='nextidx').upscale(10) # works only for D8/LDD
        d8.upscale(10, uparea=np.ones(d8.shape).ravel()) # wrong uparea shape
    try:
        d8.subconnect(d8_lr, idxout)
    except:
         pytest.fail('subconnect failed')
    try:
        d8.subarea(d8_lr, idxout)
    except:
        pytest.fail('subarea failed')
    try:
        d8.subriver(d8_lr, idxout, np.ones(d8.shape))
    except:
        pytest.fail('subriver failed')
    

def test_vector(d8):
    affine = Affine(1 / 120., 0.0, 5 + 50 / 120., 0.0, -1 / 120,
                    51 + 117 / 120.)
    try:
        gdf = d8.vector(affine=affine)
    except:
        pytest.fail('basins failed')
    assert gdf.index.size == d8.ncells


# def test_flwdir_repair():
#     flwdir = FlwdirRaster(data.copy())
#     assert flwdir.isvalid()
#     lst, hasloops = flwdir_check(flwdir._data_flat, flwdir.shape)
#     assert len(lst) == 110 and not hasloops
#     flwdir.repair() # repair edges
#     lst, hasloops = flwdir_check(flwdir._data_flat, flwdir.shape)
#     assert len(lst) == 0 and not hasloops
#     assert np.all(flwdir._data == data_repaired)
#     # create loop
#     idx = 450
#     idx_us = fd.us_indices(idx, flwdir._data_flat, flwdir.shape)[0]
#     flwdir[idx] = fd.idx_to_dd(idx, idx_us, flwdir.shape)
#     lst, hasloops = flwdir_check(flwdir._data_flat, flwdir.shape)
#     assert hasloops
#     flwdir.repair() # repair loop
#     lst, hasloops = flwdir_check(flwdir._data_flat, flwdir.shape)
#     assert len(lst) == 0 and not hasloops

# def test_setup_network():
#     flwdir = FlwdirRaster(data_repaired.copy())
#     flwdir.setup_network()
#     assert flwdir._rnodes[0].dtype == np.uint32
#     assert len(flwdir._rnodes) == len(flwdir._rnodes_up) == 174
#     tot_n = np.sum([np.sum(n != np.uint32(-1)) for n in flwdir._rnodes_up]) + flwdir._rnodes[-1].size
#     assert tot_n == flwdir.size

# def test_delineate_basins():
#     flwdir = FlwdirRaster(data_repaired.copy())
#     idx = flwdir.get_pits()
#     basins, bboxs = flwdir.delineate_basins()
#     xmin, ymin = bboxs.min(axis=0)[[0,1]]
#     xmax, ymax = bboxs.max(axis=0)[[2,3]]
#     assert np.all((xmin, ymin, xmax, ymax) == flwdir.bounds)
#     assert bboxs.shape[0] == idx.size
#     assert np.all(np.unique(basins).size==idx.size) # single basin index

#     # check bboxs of main basin
#     flwdir = FlwdirRaster(data.copy())
#     basins, bboxs = flwdir.delineate_basins(idx0)
#     assert np.all(bboxs[0] == flwdir.bounds)
#     assert np.all(np.unique(basins).size==1+1) # single basin index plus background value
#     assert np.sum(basins) == 3045

# def test_basin_maps():
#     # test single basin
#     flwdir = FlwdirRaster(data.copy())
#     flwdir.setup_network(idx0)
#     basins = flwdir.basin_map()
#     assert np.all(np.unique(basins[basins!=0])==1) # single basin index
#     assert np.sum(basins) == 3045
#     np.random.seed(0)
#     idxs = np.where(basins.ravel())[0]
#     idx = np.concatenate((
#         idxs[np.random.randint(0, np.sum(basins), 18)],
#         np.array([idx0])
#         )
#     )
#     basins2 = flwdir.basin_map(idx=idx, values=idx, dtype=np.int32)
#     assert np.sum(basins2!=0) == 3045
#     assert np.unique(basins2).size == 20 # 19 subbasins + background zero
#     # with rasterio.open(r'./tests/data/basins.tif', 'w', **prof) as dst:
#     #     dst.write(basins2, 1)

# def test_stream_order():
#     #
#     flwdir = FlwdirRaster(data.copy())
#     flwdir.setup_network(idx0)
#     stream_order = flwdir.stream_order()
#     assert stream_order.dtype == np.int8
#     assert stream_order.min() == -1
#     assert np.unique(stream_order).size == 7
#     assert np.sum(stream_order>0) == 3045
#     assert np.sum(stream_order==6) == 88

#     # flwdir = FlwdirRaster(data.copy())
#     # flwdir.repair()
#     # stream_order = flwdir.stream_order()
#     # prof.update(dtype=stream_order.dtype, nodata=-1)
#     # with rasterio.open(r'./tests/data/stream_order.tif', 'w', **prof) as dst:
#     #     dst.write(stream_order, 1)

# def test_uparea():
#     # test as if metres with identity transform
#     flwdir = FlwdirRaster(data.copy(), crs=28992) #RD New - Netherlands [metres]
#     flwdir.setup_network(idx0)
#     upa = flwdir.upstream_area()
#     assert upa.dtype == np.float64
#     tot_n = np.sum([np.sum(n != np.uint32(-1)) for n in flwdir._rnodes_up]) + flwdir._rnodes[-1].size
#     # print(np.round(upa.max()*1e6,2))
#     assert np.round(upa.max()*1e6,2) == tot_n == 3045.00
#     # test in latlon with identity transform
#     flwdir = FlwdirRaster(data.copy())
#     flwdir.setup_network(idx0)
#     upa = flwdir.upstream_area()
#     assert np.round(upa.max(),4) == 31610442.7120
#     # prof.update(dtype=upa.dtype, nodata=-9999)
#     # with rasterio.open(r'./tests/data/upa_numpy.tif', 'w', **prof) as dst:
#     #     dst.write(upa, 1)

# def test_riv_shape():
#     flwdir = FlwdirRaster(data.copy(), crs=crs, transform=transform)
#     flwdir.setup_network(idx0)
#     gdf_riv, gdf_pits = flwdir.stream_shape()
#     assert np.all([g.is_valid for g in gdf_riv.geometry])
#     assert len(gdf_riv) == 61
#     assert len(gdf_pits) == 1
#     # test with given coordinates
#     gdf_riv1, gdf_pit1 = flwdir.stream_shape(xs=np.ones(flwdir.shape), ys=np.ones(flwdir.shape))
#     assert np.all(gdf_riv1.total_bounds==1)
#     assert np.all(gdf_pit1.total_bounds==1)

# def test_upscale():
#     with rasterio.open(r'./tests/data/uparea.tif', 'r') as src:
#         uparea = src.read(1)
#         transform = src.transform
#         crs = src.crs
#     with rasterio.open(r'./tests/data/flwdir2.tif', 'r') as src:
#         data2 = src.read(1)
#     flwdir = FlwdirRaster(data_repaired.copy(), crs=crs, transform=transform, copy=True)
#     flwdir[idx0] = np.uint8(0)
#     flwdir2, outlets, cat_idx, riv_idx = flwdir.upscale(2, uparea=uparea, upa_min=0.5, return_subcatch_indices=True)
#     assert np.all(flwdir2._data == data2)
#     assert outlets.size == np.unique(outlets).size

#     # # test with large data (local only)
#     # with rasterio.open(r'./tests/data/s05w050_dir.tif', 'r') as src:
#     #     data0 = src.read(1)
#     #     transform = src.transform
#     #     crs = src.crs
#     # with rasterio.open(r'./tests/data/s05w050_upa.tif', 'r') as src:
#     #     uparea = src.read(1)
#     # with rasterio.open(r'./tests/data/s05w050_dir_05min.tif', 'r') as src:
#     #     data2 = src.read(1)
#     # flwdir = FlwdirRaster(data0, crs=crs, transform=transform)

#     # flwdir2, outlets = flwdir.upscale(100, uparea=uparea, upa_min=1.)
#     # assert np.all(flwdir2._data == data2)
#     # assert outlets.size == np.unique(outlets).size # make sure all outlets are unique

#     # xs, ys = flwdir._xycoords()
#     # gdf_riv, gdf_pits = flwdir2.stream_shape(min_order=1)
#     # gdf_riv.to_file('./tests/data/s05w050_rivers_05min_lr.shp')
#     # gdf_pits.to_file('./tests/data/s05w050_pits_05min_lr.shp')
#     # gdf_riv, gdf_pits = flwdir2.stream_shape(outlet_lr=outlets, xs=xs, ys=ys, min_order=1)
#     # gdf_riv.to_file('./tests/data/s05w050_rivers_05min.shp')
#     # gdf_pits.to_file('./tests/data/s05w050_pits_05min.shp')

#     # assert flwdir2._data.size*4 == flwdir._data.size
#     # # assert np.all(flwdir2._data == data2)
#     # # stream_order = flwdir2.stream_order()
#     # import pdb; pdb.set_trace()
#     # flwdir2 = flwdir
#     # prof = dict(
#     #     driver='GTiff',
#     #     dtype=flwdir2._data.dtype,
#     #     # dtype=upa.dtype,
#     #     height=flwdir2.shape[0],
#     #     width=flwdir2.shape[1],
#     #     transform=flwdir2.transform,
#     #     crs=flwdir2.crs,
#     #     nodata=247,
#     #     # nodata=-9999,
#     #     count=1,
#     # )
#     # with rasterio.open(r'./tests/data/s05w050_dir_05min.tif', 'w', **prof) as dst:
#     #     dst.write(flwdir2._data, 1)

# def test_dem_adjust():
#     # option 1 fill
#     dem0 = np.array([8, 7, 6, 5, 5, 6, 5, 4])
#     dem1 = np.array([8, 7, 6, 5, 5, 5, 5, 4])
#     assert np.all(pyflwdir.dem._fix_pits_streamline(dem0) ==  dem1)
#     # option 2 fill
#     dem0 = np.array([8, 7, 6, 5, 6, 6, 5, 4])
#     dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 4])
#     assert np.all(pyflwdir.dem._fix_pits_streamline(dem0) ==  dem1)
#     # option 3 dig and fill
#     dem0 = np.array([8, 7, 6, 5, 6, 7, 5, 4])
#     dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 4])
#     assert np.all(pyflwdir.dem._fix_pits_streamline(dem0) ==  dem1)
#     # TODO test full scale with small data
#     with rasterio.open(r'./tests/data/tmp/acara_620000004_30sec/flwdir.tif', 'r') as src:
#         data = src.read(1)
#         transform = src.transform
#         crs = src.crs
#     with rasterio.open(r'./tests/data/tmp/acara_620000004_30sec/outelv.tif', 'r') as src:
#         elevtn = src.read(1)
#         prof = src.profile
#     flwdir = FlwdirRaster(data, transform=transform, crs=crs)
#     elevtn_new = flwdir.adjust_elevation(elevtn, copy=True)
#     assert np.sum(elevtn!=elevtn_new) == 12

# def test_ucat_map():
#     with rasterio.open(r'./tests/data/tmp/acara_620000004_30sec/flwdir.tif', 'r') as src:
#         data = src.read(1)
#         transform = src.transform
#         crs = src.crs
#     with rasterio.open(r'./tests/data/tmp/acara_620000004_30sec/uparea.tif', 'r') as src:
#         uparea = src.read(1)
#     flwdir = FlwdirRaster(data, transform=transform, crs=crs)
#     basin, outlet = flwdir.ucat_map(10, uparea=uparea, upa_min=0.5)
#     # TODO add test

# def check_memory_time():
#     # data = xr.open_dataset(r'd:\work\flwdir_scaling\03sec\test_sel_idx74.nc')['dir'].load().values
#     data = xr.open_dataset(r'/media/data/hydro_merit_1.0/03sec/test_sel_idx74.nc')['dir'].load().values
#     idx0 = np.uint32(8640959)
#     print(rtsys.get_allocation_stats())

#     print('initialize')
#     flwdir = FlwdirRaster(data)
#     print(rtsys.get_allocation_stats())

#     print('setup network')
#     start = time.time()
#     pyflwdir.network.setup_dd(np.asarray([idx0], dtype=np.uint32), flwdir._data_flat, flwdir.shape)
#     end = time.time()
#     print(f"Elapsed (before compilation) = {(end - start):.6f} s")
#     print(rtsys.get_allocation_stats())
#     for _ in range(3):
#         start = time.time()
#         pyflwdir.network.setup_dd(np.asarray([idx0], dtype=np.uint32), flwdir._data_flat, flwdir.shape)
#         end = time.time()
#         print(f"Elapsed (after compilation) = {(end - start):.6f} s")
#         print(rtsys.get_allocation_stats())

#     # print('basin delineation')
#     # basins = flwdir.basin_map()
#     # print(rtsys.get_allocation_stats())

#     # print('basin bouhds')
#     # bounds = flwdir.delineate_basins()
#     # print(rtsys.get_allocation_stats())

#     # print('upastream area')
#     # upa = flwdir.upstream_area()
#     # print(rtsys.get_allocation_stats())

#     # print('stream oder')
#     # stro = flwdir.stream_order()
#     # print(rtsys.get_allocation_stats())

# if __name__ == "__main__":
#     # check_memory_time()
#     # print('finalize')
#     # print(rtsys.get_allocation_stats())
#     # test_flwdir_repair()
#     # test_setup_network()
#     # test_delineate_basins()
#     # test_basin_maps()
#     # test_uparea()
#     # test_stream_order()
#     # test_riv_shape()
#     # test_upscale()
#     # test_dem_adjust()
#     test_ucat_map()
#     print('success')

# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Tests for the pyflwdir module.
"""
import pytest
import numpy as np
from affine import Affine

import pyflwdir
from pyflwdir.core import _mv
from pyflwdir.gis_utils import IDENTITY

# read and parse data
@pytest.fixture
def d8_data():
    data = np.fromfile(r"./data/d8.bin", dtype=np.uint8)
    return data.reshape((678, 776))


@pytest.fixture
def ldd_data():
    data = np.fromfile(r"./data/ldd.bin", dtype=np.uint8)
    return data.reshape((678, 776))


@pytest.fixture
def nextxy_data():
    data = np.fromfile(r"./data/nextxy.bin", dtype=np.int32)
    return data.reshape((2, 678, 776))


@pytest.fixture
def nextidx_data():
    data = np.fromfile(r"./data/nextidx.bin", dtype=np.uint32)
    return data.reshape((678, 776))


@pytest.fixture
def d8():
    flw = pyflwdir.load(r"./data/flw.pkl")
    res, west, north = 1 / 120, 5 + 50 / 120.0, 51 + 117 / 120.0
    transform = Affine(res, 0.0, west, 0.0, -res, north)
    flw.set_transform(transform, latlon=True)
    flw._tree  # initialize us array
    return flw


# test object
def test_from_array(d8_data, ldd_data, nextxy_data, nextidx_data):
    ftypes = {
        "d8": d8_data,
        "ldd": ldd_data,
        "nextxy": nextxy_data,
        "nextidx": nextidx_data,
    }
    for name, fdata in ftypes.items():
        fd = pyflwdir.from_array(fdata)
        assert fd.ftype == name
        assert fd.isvalid
    # make sure these raise errors
    with pytest.raises(ValueError):
        pyflwdir.from_array(ldd_data, ftype="d8", check_ftype=True)  # invalid type
        pyflwdir.from_array(
            nextxy_data, ftype="d8", check_ftype=True
        )  # invalid shape & type
        pyflwdir.from_array(
            np.arange(20, dtype=np.uint32), ftype="infer", check_ftype=False
        )  # unknown type
        pyflwdir.from_array(
            np.array([2]), ftype="d8", check_ftype=True
        )  # invalid data: too small
        pyflwdir.from_array(d8_data, ftype="d8", transform=(0, 0))  # invalid transform


def test_attrs(d8, d8_data):
    assert d8.size == d8_data.size
    assert d8.shape == d8_data.shape
    npits = d8._idxs_pit.size
    nus = d8._idxs_us[d8._idxs_us != _mv].size
    nds = d8._idxs_ds.size
    assert nus + npits == nds
    assert d8.pits.size == 1
    assert isinstance(d8.transform, Affine)
    assert isinstance(d8.latlon, bool)
    assert isinstance(d8.pit_coords, tuple)


def test_toarray(d8, d8_data):
    assert np.all(d8.to_array() == d8_data)


# NOTE tmpdir is predefined fixture
def test_save(d8, tmpdir):
    fn = tmpdir.join("flw.pkl")
    d8.dump(fn)
    flw = pyflwdir.load(fn)
    for key in d8._dict:
        assert np.all(d8._dict[key] == flw._dict[key])


def test_stream_order(d8):
    strord = d8.stream_order()
    assert strord.flat[d8._idxs_dense].min() == 1
    assert strord.min() == -1
    assert strord.max() == 9  # NOTE specific to data
    assert strord.dtype == np.int8
    assert np.all(strord.shape == d8.shape)


def test_set_pits(d8):
    idx0 = d8.pits
    idx1 = d8._idxs_dense[0]
    x, y = d8.pit_coords
    # all cells are True -> pit at idx1
    d8.set_pits(idxs_pit=idx1, streams=np.full(d8.shape, True, dtype=np.bool))
    assert np.all(d8.pits == idx1)
    assert d8._tree_ is None
    # original pit idx0
    d8.set_pits(xy_pit=(x, y))
    assert np.all(d8.pits == idx0)
    d8.set_pits()
    assert np.all(d8.pits == idx0)
    # no streams -> pit at idx0
    d8.set_pits(idxs_pit=idx1, streams=np.full(d8.shape, False, dtype=np.bool))
    assert np.all(d8.pits == idx0)
    assert np.all(
        np.asarray(d8.pit_coords).ravel().round(6) == np.asarray([x[0], y[0]]).round(6)
    )


def test_upstream_area(d8):
    # test with upstream grid cells
    uparea = d8.upstream_area()
    assert uparea.min() == -9999
    assert uparea[uparea != -9999].min() == 1
    assert uparea.max() == d8.ncells  # NOTE specific to data with single pit
    assert uparea.dtype == np.int32
    assert np.all(uparea.shape == d8.shape)
    # compare with accuflux
    assert np.all(d8.accuflux(np.ones(d8.shape)) == uparea)
    # test upstream area in km2
    uparea2 = d8.upstream_area(unit="km2")  # km2
    assert uparea2.dtype == np.float64
    assert uparea2.max().round(0) == 158957.0
    with pytest.raises(ValueError):
        d8.upstream_area(unit="km")  # invalid unit


def test_basins(d8):
    basins = d8.basins(idxs_pit=d8.pits)
    assert basins.min() == 0
    assert basins.max() == 1  # NOTE specific to data with single pit
    assert basins.sum() == d8.ncells  # NOTE specific to data with single pit
    assert basins.dtype == np.uint32
    assert np.all(basins.shape == d8.shape)
    with pytest.raises(ValueError):
        d8.basins(ids=np.arange(2))
        d8.basins(ids=np.array([0]))


def test_basin_bounds(d8):
    d8.transform = IDENTITY
    df = d8.basin_bounds(basins=np.ones(d8.shape, dtype=np.uint32))
    assert np.all(df.loc[0, ["ymax", "xmax"]].values == d8.shape)
    idxs = d8._idxs_dense[d8._tree[30]]
    df = d8.basin_bounds(idxs_pit=idxs)
    assert df.index.size == idxs.size + 1
    with pytest.raises(ValueError):
        d8.basin_bounds(basins=np.ones((10, 10)))


def test_upscale(d8, nextidx_data):
    with pytest.raises(ValueError):
        d8.upscale(10, method="unknown")  # unknown method
        # works only for D8/LDD
        pyflwdir.from_array(nextidx, ftype="nextidx").upscale(10)
        d8.upscale(10, uparea=np.ones(d8.shape).ravel())  # wrong uparea shape
    d8_lr, idxout = d8.upscale(10)
    assert d8_lr.transform[0] == 10 * d8.transform[0]
    subcon = d8.subconnect(d8_lr, idxout)
    assert subcon.flat[d8_lr._idxs_dense].min() == 0
    assert subcon.flat[d8_lr._idxs_dense].max() == 1
    assert np.all(subcon[subcon < 0] == 255)
    subgrd = d8.subarea(d8_lr, idxout)
    assert subgrd.flat[d8_lr._idxs_dense].min() > 0
    rivlen, rivslp = d8.subriver(d8_lr, idxout, np.ones(d8.shape))
    assert rivlen.flat[d8_lr._idxs_dense].min() >= 0  # only zeros at boundary
    assert rivslp.flat[d8_lr._idxs_dense].min() >= 0


def test_vectorize(d8):
    gdf = d8.vectorize()
    assert gdf.index.size == d8.ncells
    mask = d8.stream_order() >= 3
    gdf = d8.vectorize(mask=mask)
    assert gdf.index.size == np.sum(mask)
    with pytest.raises(ValueError):
        d8.vectorize(xs=np.arange(3), ys=np.arange(3))


def test_repair(d8):
    # TODO
    pass


def test_downstream(d8):
    idx0 = d8._idxs_dense[d8._tree[12][-1]]
    mask = np.full(d8.shape, False, np.bool)
    assert np.all(d8.snap_downstream(idx0, mask)[0] == d8.pits)
    assert np.all(d8.trace_downstream(idx0, mask)[0][0][-1] == d8.pits)
    assert d8.trace_downstream(idx0, max_length=0)[0][0][-1] == idx0
    assert d8.trace_downstream(idx0, max_length=0)[1][0] == 0
    with pytest.raises(ValueError):
        d8.snap_downstream(idx0, np.arange(2))
        d8.trace_downstream(idx0, np.arange(2))
        d8.trace_downstream(idx0, unit="wrong")
        d8.trace_downstream(idx0, unit="wrong")


def test_pfafstetter(d8):
    pfaf = d8._sparsify(d8.pfafstetter(min_upa=0, depth=1))
    assert np.all(np.unique(pfaf) == np.arange(1, 10))
    uparea = d8.upstream_area()
    pfaf = d8._sparsify(d8.pfafstetter(uparea=uparea, min_upa=0, depth=2))
    assert np.unique(pfaf).size == 73  # basins specific
    pfaf = d8._sparsify(d8.pfafstetter(uparea=uparea, min_upa=1e3, depth=2))
    assert np.unique(pfaf).size == 67  # basins specific


def test_dem_adjust(d8):
    elevtn = np.ones(d8.shape)
    idxs0 = d8._idxs_dense[d8._tree[30]]
    elevtn.flat[idxs0] = 2.0  # create values that need fix
    elevtn_new = d8.adjust_elevation(elevtn)
    assert np.all(elevtn_new.flat[idxs0] == 1.0)
    ndiff = np.sum(elevtn_new.flat[d8._idxs_dense] != elevtn.flat[d8._idxs_dense])
    assert ndiff == idxs0.size


def test_moving_average(d8):
    data = np.random.random(d8.shape)
    weigths = np.ones(d8.shape)
    data_smooth = d8.moving_average(data, weigths, 1)
    assert data_smooth.flat[d8._idxs_dense].max() < data.flat[d8._idxs_dense].max()
    assert np.all(data_smooth.flat[d8._idxs_dense] != data.flat[d8._idxs_dense])
    idx0 = d8._tree[-1][0]
    idxs = d8._idxs_dense[np.array([idx0, d8._idxs_ds[idx0]])]
    assert np.isclose(np.mean(data.flat[idxs]), data_smooth.flat[idxs[0]])

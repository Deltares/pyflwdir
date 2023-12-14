# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.dem module."""

import pytest
import numpy as np
from pyflwdir import dem, subgrid
from test_core import test_data

parsed, flwdir = test_data[0]
idxs_ds, idxs_pit, seq, rank, mv = parsed


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_from_dem(dtype):
    # example from Wang & Lui (2015)
    a = np.array(
        [
            [15, 15, 14, 15, 12, 6, 12],
            [14, 13, 10, 12, 15, 17, 15],
            [15, 15, 9, 11, 8, 15, 15],
            [16, 17, 8, 16, 15, 7, 5],
            [19, 18, 19, 18, 17, 15, 14],
        ],
        dtype=dtype,
    )
    # NOTE: compared to paper same a_filled, but difference
    # in flowdir because row instead of col first ..
    # and priority of non-boundary cells with same elevation
    d8 = np.array(
        [
            [2, 2, 4, 8, 1, 0, 16],
            [1, 1, 2, 2, 128, 64, 32],
            [128, 128, 1, 1, 2, 2, 4],
            [128, 128, 128, 128, 1, 1, 0],
            [64, 128, 64, 32, 128, 128, 64],
        ],
        dtype=np.uint8,
    )
    # test default
    a2 = a.copy()
    a2[1:4, 2] = 11  # filled depression
    a_filled, _d8 = dem.fill_depressions(a)
    assert np.all(a_filled == a2)
    assert np.all(d8 == _d8)
    # test single outlet
    a2[0, 5] = 12
    a_filled = dem.fill_depressions(a, outlets="min")[0]
    assert np.all(a2 == a_filled)
    # test with nodata values
    a2 = a.copy()
    a2[3, 5:] = -9999
    _d8 = dem.fill_depressions(a2)[1]
    assert np.all(_d8[3, 5:] == 247)
    assert _d8[2, 4] == 0
    # test max depth
    _a = dem.fill_depressions(a, max_depth=2)[0]
    assert np.all(a == _a)
    # test with 4-connectivity
    a2 = np.array(
        [
            [15, 15, 14, 15, 12, 6, 12],
            [14, 14, 14, 14, 15, 17, 15],
            [15, 15, 14, 14, 14, 15, 15],
            [16, 17, 14, 16, 15, 7, 5],
            [19, 18, 19, 18, 17, 15, 14],
        ],
        dtype=np.float32,
    )
    a_filled, _d8 = dem.fill_depressions(a, connectivity=4)
    assert np.all(a_filled == a2)
    assert np.all(np.isin(np.unique(_d8), [0, 1, 4, 16, 64]))


def test_dem_adjust():
    # option 1 dig
    dem0 = np.array([8, 7, 6, 5, 5, 6, 5, 4])
    dem1 = np.array([8, 7, 6, 5, 5, 5, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # option 2 fill
    dem0 = np.array([8, 7, 6, 5, 6, 6, 5, 4])
    dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # option 3 dig and fill
    dem0 = np.array([8, 7, 6, 5, 6, 7, 5, 4])
    dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    dem0 = np.array([84, 19, 5, 26, 34, 4])
    dem1 = np.array([84, 26, 26, 26, 26, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    dem0 = np.array([46, 26, 5, 20, 23, 21, 5])
    dem1 = np.array([46, 26, 21, 21, 21, 21, 5])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # with large z on last position
    dem0 = np.array([8, 7, 6, 6, 6, 6, 5, 7])
    dem1 = np.array([8, 7, 7, 7, 7, 7, 7, 7])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # pit at first value
    dem0 = np.array([5, 41, 15])
    dem1 = np.array([15, 15, 15])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # multiple pits
    dem0 = np.array([60, 13, 54, 37, 49, 27, 22, 19, 42, 33, 40, 36, 7, 32, 8, 8, 2, 1])
    dem1 = np.array([60, 54, 54, 37, 37, 27, 22, 19, 19, 19, 19, 19, 8, 8, 8, 8, 2, 1])
    assert np.all(dem._adjust_elevation(dem0) == dem1)


# TODO: extend test
def test_slope():
    elv = np.ones((4, 4))
    nodata = -9999
    assert np.all(dem.slope(elv, nodata) == 0)
    elv.flat[0] == -9999
    assert np.all(dem.slope(elv, nodata).flat[1:] == 0)
    elv = np.random.random((4, 4))
    assert np.all(dem.slope(elv, nodata).flat[1:] >= 0)
    elv = np.random.random((1, 1))
    assert np.all(dem.slope(elv, nodata) == 0)


def test_hand_fldpln():
    elevtn = rank  # dz along flow path is 1
    drain = rank == 0  # only outlets are
    # hand == elevtn
    hand = dem.height_above_nearest_drain(idxs_ds, seq, drain, elevtn)
    assert np.all(hand == elevtn)
    # subgrid floodplain volume
    idxs_out = np.where(drain.ravel())[0]
    area = np.ones(idxs_ds.size, dtype=np.int32)
    depths = np.linspace(1, hand.max(), 5)
    drain_map, fldpln_vol = subgrid.ucat_volume(
        idxs_out, idxs_ds, seq, hand, area, depths=depths
    )
    assert fldpln_vol.shape == (*depths.shape, drain_map.max())
    assert np.all(np.diff(fldpln_vol, axis=0) > 0)
    # max h = 1
    uparea = np.where(drain, 1, 0)
    upa_min = 1
    fldpln = dem.floodplains(idxs_ds, seq, elevtn, uparea, upa_min=upa_min, b=0)
    assert np.all(fldpln[elevtn > 1] == 0)
    assert np.all(fldpln[elevtn <= 1] != 0)
    # max h = uparea
    fldpln = dem.floodplains(idxs_ds, seq, elevtn, uparea, upa_min=upa_min, b=1)
    hmax = uparea[drain].max()
    hmin = uparea[drain].min()
    assert np.all(fldpln[elevtn > hmax] == 0)
    assert np.all(fldpln[elevtn < hmin] != 0)
    # hand == 1 for non-drain cells
    elevtn = np.ones(flwdir.size)
    elevtn[drain] = 0
    hand = dem.height_above_nearest_drain(idxs_ds, seq, drain, elevtn)
    assert np.all(hand[rank > 0] == 1)
    # fldpln == 1 for all cells
    fldpln = dem.floodplains(idxs_ds, seq, elevtn, uparea, upa_min=upa_min, b=1)
    assert np.all(fldpln[rank >= 0] == 1)

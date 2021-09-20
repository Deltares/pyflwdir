# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.dem module."""

import pytest
import numpy as np
from pyflwdir import dem
from test_core import test_data

parsed, flwdir = test_data[0]
idxs_ds, idxs_pit, seq, rank, mv = parsed


def test_from_dem():
    # example from Wang & Lui (2015)
    a = np.array(
        [
            [15, 15, 14, 15, 12, 6, 12],
            [14, 13, 10, 12, 15, 17, 15],
            [15, 15, 9, 11, 8, 15, 15],
            [16, 17, 8, 16, 15, 7, 5],
            [19, 18, 19, 18, 17, 15, 14],
        ],
        dtype=np.float32,
    )
    # NOTE: compared to paper same a_filled, but difference
    # in flowdir because row instead of col first ..
    d8 = np.array(
        [
            [2, 2, 4, 8, 1, 0, 16],
            [1, 1, 2, 2, 128, 64, 32],
            [128, 128, 1, 1, 2, 2, 4],
            [64, 128, 128, 128, 1, 1, 0],
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
    a2 = np.array(
        [
            [15, 15, 14, 15, 12, 15, 17.0],
            [14, 13, 11, 12, 15, 17, 15.0],
            [15, 15, 11, 11, 8, 15, 15.0],
            [16, 17, 11, 16, 15, 7, 5.0],
            [19, 18, 19, 18, 17, 15, 14.0],
        ],
        dtype=np.float32,
    )
    a_filled = dem.fill_depressions(a, outlets="min")[0]
    assert np.all(a2 == a_filled)
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
    # test with nodata values
    a[3, 5:] = -9999
    _d8 = dem.fill_depressions(a)[1]
    assert np.all(_d8[3, 5:] == 247)
    assert _d8[2, 4] == 0
    # test max depth
    _a = dem.fill_depressions(a, max_depth=2)[0]
    assert np.all(a == _a)


def test_dem_adjust():
    # # option 1 fill
    dem0 = np.array([8, 7, 6, 5, 5, 6, 5, 4])
    dem1 = np.array([8, 7, 6, 5, 5, 5, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # # option 2 fill
    dem0 = np.array([8, 7, 6, 5, 6, 6, 5, 4])
    dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # option 3 dig and fill
    dem0 = np.array([8, 7, 6, 5, 6, 7, 5, 4])
    dem1 = np.array([8, 7, 6, 6, 6, 6, 5, 4])
    assert np.all(dem._adjust_elevation(dem0) == dem1)
    # with zmax on last position
    dem0 = np.array([8, 7, 6, 6, 6, 6, 5, 6])
    dem1 = np.array([8, 7, 6, 6, 6, 6, 6, 6])
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

# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.dem module."""

import pytest
import numpy as np
from pyflwdir import dem, streams
from test_core import test_data

parsed, flwdir = test_data[0]
idxs_ds, idxs_pit, seq, rank, mv = parsed


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
    nodata = -9999.0
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

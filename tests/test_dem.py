# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
"""Tests for the pyflwdir module.
"""
import pytest
import numpy as np

import pyflwdir
from pyflwdir.core import _mv
from pyflwdir import dem


def test_dem_adjust():
    # option 1 fill
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
    # TODO test full scale with small data
    # with rasterio.open(r'./tests/data/tmp/acara_620000004_30sec/flwdir.tif', 'r') as src:
    #     data = src.read(1)
    #     transform = src.transform
    #     crs = src.crs
    # with rasterio.open(r'./tests/data/tmp/acara_620000004_30sec/outelv.tif', 'r') as src:
    #     elevtn = src.read(1)
    #     prof = src.profile
    # flwdir = FlwdirRaster(data, transform=transform, crs=crs)
    # elevtn_new = flwdir.adjust_elevation(elevtn, copy=True)
    # assert np.sum(elevtn!=elevtn_new) == 12

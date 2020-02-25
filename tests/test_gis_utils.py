#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the pyflwdir module.
"""
import pytest

import numpy as np

import pyflwdir
from pyflwdir import gis_utils

# glob total area
glob_area = 4 * np.pi * gis_utils._R ** 2

# reggrid_dx(lats, lons)
# reggrid_dy(lats, lons)
# distance(lon1, lat1, lon2, lat2)
# degree_metres_y(lat)
# degree_metres_x(lat)


def test_cellarea():
    # area of whole sphere
    assert gis_utils.cellarea(0, 360, 180) == glob_area
    # area of 1 degree cell
    assert gis_utils.cellarea(0, 1, 1) == 12364154779.389229


def test_reggrid_area():
    # area of glob in 1 degree cells
    lats = np.arange(-89.5, 90)
    lons = np.arange(-179.5, 180)
    assert gis_utils.reggrid_area(lats, lons).sum().round() == np.round(glob_area)


def test_idxs_to_coords():
    shape = (10, 8)
    idxs_valid = np.arange(shape[0] * shape[1]).reshape(shape)
    affine = gis_utils.IDENTITY
    xs, ys = gis_utils.idxs_to_coords(idxs_valid, affine, shape)
    assert np.all(ys == (np.arange(shape[0]) + 0.5)[:, None])
    assert np.all(xs == np.arange(shape[1]) + 0.5)


def test_affine_to_coords():
    shape = (10, 8)
    affine = gis_utils.IDENTITY
    xs, ys = gis_utils.affine_to_coords(affine, shape)
    assert np.all(ys == np.arange(shape[0]) + 0.5)
    assert np.all(xs == np.arange(shape[1]) + 0.5)

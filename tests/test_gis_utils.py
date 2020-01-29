#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the pyflwdir module.
"""
import pytest

import numpy as np

import pyflwdir
from pyflwdir import gis_utils

# reggrid_dx(lats, lons)
# reggrid_dy(lats, lons)
# reggrid_area(lats, lons)
# cellarea(lat, xres, yres)
# distance(lon1, lat1, lon2, lat2)
# degree_metres_y(lat)
# degree_metres_x(lat)


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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the pyflwdir module.
"""
import pytest

import numpy as np
from rasterio.transform import Affine

import pyflwdir
from pyflwdir import gridtools


def test_transform_conv():
    shape = (10, 20)
    transform = Affine(1, 0, 0, 0, -0.5, 5)
    # NOTE: will high resolution we don't get the same return
    # transform = Affine(1/120., 0, 0, 0, -1/1200., 5) 
    lat, lon = gridtools.transform_to_latlon(transform, shape)
    transform2 = gridtools.latlon_to_transform(lat, lon)
    assert np.all(transform2 == transform)


def test_res_area():
    res = 1/1200.
    shape = (3,4)
    transform = Affine(res, 0, 0, 0, -res, 0)
    dy, dx = gridtools.latlon_cellres_metres(transform, shape)
    assert np.all(dy.shape == dx.shape == (shape[0],))
    assert np.round(dy[0],6)==92.145227 and np.round(dx[0],6)==92.766215
    cellare = gridtools.latlon_cellare_metres(transform, shape)
    assert np.all(cellare.shape == shape)
    assert np.round(cellare[0,0], 8) == 8547.96396208
    
    transform = Affine(res, 0, 0, 0, -res, 90)
    dy, dx = gridtools.latlon_cellres_metres(transform, (1, 1))
    assert np.all(np.round(dx,2)==0)

    transform = Affine(1, 0, 0, 0, -0.5, 0)
    cellare = gridtools.cellare_metres(transform, shape)
    assert np.all(cellare.shape == shape)
    assert np.all(np.round(cellare[0,0],8)==0.5)

if __name__ == "__main__":
    test_transform_conv()
    test_res_area()
    import pdb; pdb.set_trace()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the pyflwdir module.
"""
import pytest

import numpy as np
import rasterio
from rasterio.transform import Affine

import pyflwdir
from pyflwdir import gridtools

with rasterio.open(r'./tests/data/basins.tif', 'r') as src:
    raster = src.read(1)
    nodata = src.nodata
    transform = src.transform
    crs = src.crs
    prof = src.profile

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
    assert np.all(cellare.shape == (shape[0],))
    assert np.round(cellare[0], 8) == 8547.96396208
    
    transform = Affine(res, 0, 0, 0, -res, 90)
    dy, dx = gridtools.latlon_cellres_metres(transform, (1, 1))
    assert np.all(np.round(dx,2)==0)

def test_vectorize():
    # TODO improve test
    gridtools.vectorize(raster, nodata, transform, crs=crs)

def test_idx_to_xy():
    nrow, ncol = 2, 5
    idx = np.arange(nrow*ncol, dtype=np.int64).reshape((nrow, ncol))
    xs = np.arange(1,ncol+1, dtype=np.float64)
    ys = np.arange(1,nrow+1, dtype=np.float64)
    x, y = gridtools.idx_to_xy(idx, xs, ys, ncol)
    assert np.all(x.shape == y.shape == (nrow, ncol))
    assert np.all(x[0,:].ravel() == xs)
    assert np.all(y[:,0].ravel() == ys)

if __name__ == "__main__":
    # test_transform_conv()
    test_res_area()
    # test_vectorize()

    pass
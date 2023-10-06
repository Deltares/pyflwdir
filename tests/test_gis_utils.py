# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.gis_utils module."""

import pytest
import numpy as np
from pyflwdir import gis_utils as gis
from affine import Affine

# glob total area
glob_area = 4 * np.pi * gis._R**2
glob_circ = 2 * np.pi * gis._R


## TRANSFORM
# Adapted from https://github.com/mapbox/rasterio/blob/master/tests/test_transform.py
prof = {
    "width": 6120,
    "height": 4920,
    "res": 1 / 1200.0,
    "bounds": [-10.5, 51.4, -5.4, 55.5],
    "transform": Affine(1 / 1200.0, 0.0, -10.5, 0.0, -1 / 1200.0, 55.5),
}


def test_from_origin():
    w, _, _, n = prof["bounds"]
    tr = gis.transform_from_origin(w, n, prof["res"], prof["res"])
    assert [round(v, 7) for v in tr] == [round(v, 7) for v in prof["transform"]]


def test_from_bounds():
    w, s, e, n = prof["bounds"]
    tr = gis.transform_from_bounds(w, s, e, n, prof["width"], prof["height"])
    assert [round(v, 7) for v in tr] == [round(v, 7) for v in prof["transform"]]


def test_array_bounds():
    bounds0 = np.asarray(prof["bounds"])
    bounds = gis.array_bounds(prof["height"], prof["width"], prof["transform"])
    assert np.all(bounds0 == np.asarray(bounds).round(7))


def test_xy():
    aff = prof["transform"]
    ul_x, ul_y = aff * (0, 0)
    xoff = aff.a
    yoff = aff.e
    assert gis.xy(aff, 0, 0, offset="ul") == (ul_x, ul_y)
    assert gis.xy(aff, 0, 0, offset="ur") == (ul_x + xoff, ul_y)
    assert gis.xy(aff, 0, 0, offset="ll") == (ul_x, ul_y + yoff)
    expected = (ul_x + xoff, ul_y + yoff)
    assert gis.xy(aff, 0, 0, offset="lr") == expected
    expected = (ul_x + xoff / 2, ul_y + yoff / 2)
    assert gis.xy(aff, 0, 0, offset="center") == expected
    assert (
        gis.xy(aff, 0, 0, offset="lr")
        == gis.xy(aff, 0, 1, offset="ll")
        == gis.xy(aff, 1, 1, offset="ul")
        == gis.xy(aff, 1, 0, offset="ur")
    )


def test_rowcol():
    aff = gis.IDENTITY  # N->S changed in version 0.5
    left, bottom, right, top = (0, -200, 100, 0)
    assert gis.rowcol(aff, left, top) == (top, left)
    assert gis.rowcol(aff, right, top) == (top, right)
    assert gis.rowcol(aff, right, bottom) == (-bottom, right)
    assert gis.rowcol(aff, left, bottom) == (-bottom, left)


def test_idxs_to_coords():
    shape = (10, 8)
    idxs = np.arange(shape[0] * shape[1]).reshape(shape)
    transform = gis.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    xs, ys = gis.idxs_to_coords(idxs, transform, shape)
    assert np.all(ys == (np.arange(shape[0]) + 0.5)[:, None])
    assert np.all(xs == np.arange(shape[1]) + 0.5)
    with pytest.raises(IndexError):
        gis.idxs_to_coords(np.array([-1]), transform, shape)


def test_coords_to_idxs():
    shape = (10, 8)
    idxs0 = np.arange(shape[0] * shape[1])
    transform = gis.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    xs, ys = np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)
    idxs = gis.coords_to_idxs(xs, ys, transform, shape)
    assert np.all(idxs.ravel() == idxs0)
    with pytest.raises(IndexError):
        gis.coords_to_idxs(ys, xs, transform, shape)


def test_affine_to_coords():
    shape = (10, 8)
    transform = gis.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    xs, ys = gis.affine_to_coords(transform, shape)
    assert np.all(ys == np.arange(shape[0]) + 0.5)
    assert np.all(xs == np.arange(shape[1]) + 0.5)


def test_reggrid_dx():
    # also tests degree_metres_x
    # area of glob in 1 degree cells
    lats = np.array([0.0])
    lons = np.arange(-179.5, 180)
    dx = gis.reggrid_dx(lats, lons)
    assert dx.shape == (lats.size, lons.size)
    assert dx.sum().round(3) == 40075004.88


def test_reggrid_dy():
    # also tests degree_metres_y
    # area of glob in 1 degree cells
    lats = np.arange(-89.5, 90)
    lons = np.array([0.0])
    dy = gis.reggrid_dy(lats, lons)
    assert dy.shape == (lats.size, lons.size)
    assert dy.sum().round(3) == 20003925.600


def test_cellarea():
    # area of whole sphere
    assert gis.cellarea(0, 360, 180) == glob_area
    # area of 1 degree cell
    assert gis.cellarea(0, 1, 1) == 12364154779.389229


def test_reggrid_area():
    # area of glob in 1 degree cells
    lats = np.arange(-89.5, 90)
    lons = np.arange(-179.5, 180)
    assert gis.reggrid_area(lats, lons).sum().round() == np.round(glob_area)


def test_distance():
    # transform=gis.IDENTITY
    assert gis.distance(0, 1, 3) == 1  # horizontal
    assert gis.distance(0, 3, 3) == 1  # vertical
    assert gis.distance(4, 0, 3) == np.hypot(1, 1)  # diagonal
    assert gis.distance(0, 4, 3, True) == gis.distance(4, 0, 3, True)
    assert gis.distance(0, 1, 3, False) == gis.distance(7, 8, 3, False)
    assert gis.distance(0, 1, 3, True) != gis.distance(7, 8, 3, True)


def test_edge():
    a = np.ones((5, 5), dtype=bool)
    b = a.copy()
    b[1:-1, 1:-1] = False
    assert np.all(gis.get_edge(a) == b)
    a[np.diag_indices(5)] = False
    assert np.all(gis.get_edge(a) == a)
    b = a.copy()
    b[1, 3], b[3, 1] = False, False
    d4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    assert np.all(gis.get_edge(a, structure=d4) == b)


def test_spread():
    a = np.zeros((5, 5))
    a[2, 2] = 1
    out, src, dst = gis.spread2d(a, nodata=0)
    assert np.all(out == 1)
    assert np.all(src == 12)
    assert np.isclose(np.max(dst), 2 * np.hypot(1, 1))
    a[-1, -1] = 2
    out, src, dst = gis.spread2d(a, nodata=0, msk=a != 2, latlon=True)
    assert np.all(out[a != 2] == 1)
    assert np.all(out.flat[src] == out)
    assert dst[-1, -1] == 0

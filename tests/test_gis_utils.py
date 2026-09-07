"""Tests for the pyflwdir.gis_utils module."""

import numpy as np
import pytest
from affine import Affine

from pyflwdir import gis_utils as gis

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

TRANSFORM_CASES = [
    pytest.param(
        (3, 4),
        Affine(1.0, 0.0, 100.0, 0.0, -2.0, 50.0),
        (100.0, 44.0, 104.0, 50.0),
        {
            "ul": (100.0, 50.0),
            "ur": (101.0, 50.0),
            "ll": (100.0, 48.0),
            "lr": (101.0, 48.0),
            "center": (100.5, 49.0),
        },
        np.array([100.5, 101.5, 102.5, 103.5]),
        np.array([49.0, 47.0, 45.0]),
        id="north-up",
    ),
    pytest.param(
        (3, 4),
        Affine(1.25, 0.0, -3.0, 0.0, 2.5, 4.0),
        (-3.0, 11.5, 2.0, 4.0),
        {
            "ul": (-3.0, 4.0),
            "ur": (-1.75, 4.0),
            "ll": (-3.0, 6.5),
            "lr": (-1.75, 6.5),
            "center": (-2.375, 5.25),
        },
        np.array([-2.375, -1.125, 0.125, 1.375]),
        np.array([5.25, 7.75, 10.25]),
        id="south-up",
    ),
    pytest.param(
        (3, 4),
        Affine(2.0, 0.25, 10.0, 0.5, -3.0, 20.0),
        (10.0, 13.0, 18.75, 20.0),
        {
            "ul": (10.0, 20.0),
            "ur": (12.0, 20.5),
            "ll": (10.25, 17.0),
            "lr": (12.25, 17.5),
            "center": (11.125, 18.75),
        },
        np.array([11.125, 13.125, 15.125, 17.125]),
        np.array([18.75, 15.75, 12.75]),
        id="rotated-north-up",
    ),
    pytest.param(
        (3, 4),
        Affine(1.5, -0.25, -5.0, -0.4, 2.0, 7.0),
        (-5.0, 11.4, 0.25, 7.0),
        {
            "ul": (-5.0, 7.0),
            "ur": (-3.5, 6.6),
            "ll": (-5.25, 9.0),
            "lr": (-3.75, 8.6),
            "center": (-4.375, 7.8),
        },
        np.array([-4.375, -2.875, -1.375, 0.125]),
        np.array([7.8, 9.8, 11.8]),
        id="rotated-south-up",
    ),
]


def _transform_xy(transform, cols, rows):
    xs = transform.a * cols + transform.b * rows + transform.c
    ys = transform.d * cols + transform.e * rows + transform.f
    return xs, ys


def test_from_origin():
    w, _, _, n = prof["bounds"]
    tr = gis.transform_from_origin(w, n, prof["res"], prof["res"])
    assert [round(v, 7) for v in tr] == [round(v, 7) for v in prof["transform"]]


def test_from_bounds():
    w, s, e, n = prof["bounds"]
    tr = gis.transform_from_bounds(w, s, e, n, prof["width"], prof["height"])
    assert [round(v, 7) for v in tr] == [round(v, 7) for v in prof["transform"]]


@pytest.mark.parametrize(
    "shape, transform, expected_bounds, _, __, ___", TRANSFORM_CASES
)
def test_array_bounds(shape, transform, expected_bounds, _, __, ___):
    height, width = shape
    bounds = gis.array_bounds(height, width, transform)
    assert np.allclose(bounds, expected_bounds)


@pytest.mark.parametrize(
    "shape, transform, _, expected_offsets, __, ___", TRANSFORM_CASES
)
def test_xy(shape, transform, _, expected_offsets, __, ___):
    for offset, expected in expected_offsets.items():
        assert gis.xy(transform, 0, 0, offset=offset) == expected
    assert (
        gis.xy(transform, 0, 0, offset="lr")
        == gis.xy(transform, 0, 1, offset="ll")
        == gis.xy(transform, 1, 1, offset="ul")
        == gis.xy(transform, 1, 0, offset="ur")
    )

    rows, cols = np.indices(shape)
    xs, ys = gis.xy(transform, rows, cols)
    expected = _transform_xy(transform, cols + 0.5, rows + 0.5)
    assert np.allclose(xs, expected[0])
    assert np.allclose(ys, expected[1])


def test_rowcol():
    aff = gis.IDENTITY  # N->S changed in version 0.5
    left, bottom, right, top = (0, -200, 100, 0)
    assert gis.rowcol(aff, left, top) == (top, left)
    assert gis.rowcol(aff, right, top) == (top, right)
    assert gis.rowcol(aff, right, bottom) == (-bottom, right)
    assert gis.rowcol(aff, left, bottom) == (-bottom, left)


@pytest.mark.parametrize("shape, transform, _, __, ___, ____", TRANSFORM_CASES)
def test_rowcol_transform_cases(shape, transform, _, __, ___, ____):
    rows, cols = np.indices(shape)
    xs, ys = _transform_xy(transform, cols + 0.5, rows + 0.5)
    rows1, cols1 = gis.rowcol(transform, xs, ys)
    assert np.all(rows1 == rows)
    assert np.all(cols1 == cols)


@pytest.mark.parametrize("shape, transform, _, __, ___, ____", TRANSFORM_CASES)
def test_idxs_to_coords(shape, transform, _, __, ___, ____):
    idxs = np.arange(shape[0] * shape[1]).reshape(shape)
    xs, ys = gis.idxs_to_coords(idxs, transform, shape)
    rows, cols = np.indices(shape)
    expected = _transform_xy(transform, cols + 0.5, rows + 0.5)
    assert np.allclose(xs, expected[0])
    assert np.allclose(ys, expected[1])
    with pytest.raises(IndexError):
        gis.idxs_to_coords(np.array([-1]), transform, shape)


@pytest.mark.parametrize("shape, transform, _, __, ___, ____", TRANSFORM_CASES)
def test_coords_to_idxs(shape, transform, _, __, ___, ____):
    idxs0 = np.arange(shape[0] * shape[1])
    rows, cols = np.indices(shape)
    xs, ys = _transform_xy(transform, cols + 0.5, rows + 0.5)
    idxs = gis.coords_to_idxs(xs, ys, transform, shape)
    assert np.all(idxs.ravel() == idxs0)
    with pytest.raises(IndexError):
        gis.coords_to_idxs(ys, xs, transform, shape)


@pytest.mark.parametrize(
    "shape, transform, _, __, expected_xcoords, expected_ycoords", TRANSFORM_CASES
)
def test_affine_to_coords(shape, transform, _, __, expected_xcoords, expected_ycoords):
    xs, ys = gis.affine_to_coords(transform, shape)
    assert np.allclose(xs, expected_xcoords)
    assert np.allclose(ys, expected_ycoords)


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


@pytest.mark.parametrize(
    "structure",
    [
        [[True] * 3] * 3,
        np.ones((2, 2), dtype=bool),
        np.ones((3, 3), dtype=np.uint8),
    ],
)
def test_edge_structure_validation(structure):
    with pytest.raises(ValueError, match="structure must be a 3x3 boolean array"):
        gis.get_edge(np.ones((5, 5), dtype=bool), structure=structure)


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

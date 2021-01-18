# -- coding: utf-8 --
""""""

from numba import njit
import numpy as np
import math
from affine import identity as IDENTITY
from affine import Affine

_R = 6371e3  # Radius of earth in m. Use 3956e3 for miles

__all__ = [
    "transform_from_origin",
    "transform_from_bounds",
    "array_bounds",
    "xy",
    "rowcol",
    "idxs_to_coords",
    "affine_to_coords",
    "reggrid_area",
    "reggrid_dy",
    "reggrid_dx",
]

## TRANSFORM
# Adapted from https://github.com/mapbox/rasterio/blob/master/rasterio/transform.py
# changed xy and rowcol to work directly on numpy arrays
# avoid gdal dependency


def transform_from_origin(west, north, xsize, ysize):
    """Return an Affine transformation given upper left and pixel sizes.
    Return an Affine transformation for a georeferenced raster given
    the coordinates of its upper left corner `west`, `north` and pixel
    sizes `xsize`, `ysize`.
    """
    return Affine.translation(west, north) * Affine.scale(xsize, -ysize)


def transform_from_bounds(west, south, east, north, width, height):
    """Return an Affine transformation given bounds, width and height.
    Return an Affine transformation for a georeferenced raster given
    its bounds `west`, `south`, `east`, `north` and its `width` and
    `height` in number of pixels.
    """
    return Affine.translation(west, north) * Affine.scale(
        (east - west) / width, (south - north) / height
    )


def array_bounds(height, width, transform):
    """Return the bounds of an array given height, width, and a transform.
    Return the `west, south, east, north` bounds of an array given
    its height, width, and an affine transform.
    """
    w, n = transform.xoff, transform.yoff
    e, s = transform * (width, height)
    return w, s, e, n


def xy(transform, rows, cols, offset="center"):
    """Returns the x and y coordinates of pixels at `rows` and `cols`.
    The pixel's center is returned by default, but a corner can be returned
    by setting `offset` to one of `ul, ur, ll, lr`.

    Parameters
    ----------
    transform : affine transform
        Transformation from pixel coordinates to coordinate reference system.
    rows : ndarray or int
        Pixel rows.
    cols : ndarray or int
        Pixel columns.
    offset : {'center', 'ul', 'ur', 'll', 'lr'}
        Determines if the returned coordinates are for the center of the
        pixel or for a corner.

    Returns
    -------
    xs : ndarray of float
        x coordinates in coordinate reference system
    ys : ndarray of float
        y coordinates in coordinate reference system
    """
    rows, cols = np.asarray(rows), np.asarray(cols)

    if offset == "center":
        coff, roff = (0.5, 0.5)
    elif offset == "ul":
        coff, roff = (0, 0)
    elif offset == "ur":
        coff, roff = (1, 0)
    elif offset == "ll":
        coff, roff = (0, 1)
    elif offset == "lr":
        coff, roff = (1, 1)
    else:
        raise ValueError("Invalid offset")

    xs, ys = transform * transform.translation(coff, roff) * (cols, rows)
    return xs, ys


def rowcol(transform, xs, ys, op=np.floor, precision=None):
    """
    Returns the rows and cols of the pixels containing (x, y) given a
    coordinate reference system.
    Use an epsilon, magnitude determined by the precision parameter
    and sign determined by the op function:
        positive for floor, negative for ceil.
    Parameters
    ----------
    transform : Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
    xs : ndarray or float
        x values in coordinate reference system
    ys : ndarray or float
        y values in coordinate reference system
    op : function {numpy.floor, numpy.ceil, numpy.round}
        Function to convert fractional pixels to whole numbers
    precision : int, optional
        Decimal places of precision in indexing, as in `round()`.

    Returns
    -------
    rows : ndarray of ints
        array of row indices
    cols : ndarray of ints
        array of column indices
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    if precision is None:
        eps = 0.0
    else:
        eps = 10.0 ** -precision * (1.0 - 2.0 * op(0.1))
    invtransform = ~transform
    fcols, frows = invtransform * (xs + eps, ys - eps)
    cols, rows = op(fcols).astype(int), op(frows).astype(int)
    return rows, cols


def idxs_to_coords(idxs, transform, shape, offset="center"):
    """Returns coordinates of idxs raster indices based affine.

    Parameters
    ----------
    idxs : ndarray of int
        linear indices
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.
    offset : {'center', 'ul', 'ur', 'll', 'lr'}
        Determines if the returned coordinates are for the center of the
        pixel or for a corner.

    Returns
    -------
    xs : ndarray of float
        x coordinates in coordinate reference system
    ys : ndarray of float
        y coordinates in coordinate reference system

    Raises
    ------
    IndexError
        if any linear index outside domain.
    """
    idxs = np.asarray(idxs).astype(np.int)
    size = np.multiply(*shape)
    if np.any(np.logical_or(idxs < 0, idxs >= size)):
        raise IndexError("idxs coordinates outside domain")
    ncol = shape[1]
    rows = idxs // ncol
    cols = idxs % ncol
    return xy(transform, rows, cols, offset=offset)


def coords_to_idxs(xs, ys, transform, shape, op=np.floor, precision=None):
    """Returns linear indices of coordinates.

    Parameters
    ----------
    xs : ndarray or float
        x values in coordinate reference system
    ys : ndarray or float
        y values in coordinate reference system
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.
    op : function {numpy.floor, numpy.ceil, numpy.round}
        Function to convert fractional pixels to whole numbers
    precision : int, optional
        Decimal places of precision in indexing, as in `round()`.

    Returns
    -------
    idxs : ndarray of ints
        array of linear indices

    Raises
    ------
    IndexError
        if any coordinate outside domain.
    """
    nrow, ncol = shape
    rows, cols = rowcol(transform, xs, ys, op=op, precision=precision)
    if not np.all(
        np.logical_and(
            np.logical_and(rows >= 0, rows < nrow),
            np.logical_and(cols >= 0, cols < ncol),
        )
    ):
        raise IndexError("XY coordinates outside domain")
    return rows * ncol + cols


# TODO: rename to transform_to_coords & correct upstream use in pyflwdir and hydromt
def affine_to_coords(affine, shape):
    """Returs a raster axis with pixel center coordinates based on the affine.

    Parameters
    ----------
    affine : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.

    Returns
    -------
    x, y coordinate arrays : tuple of ndarray of float
    """
    height, width = shape
    x_coords, _ = affine * (np.arange(width) + 0.5, np.zeros(width) + 0.5)
    _, y_coords = affine * (np.zeros(height) + 0.5, np.arange(height) + 0.5)
    return x_coords, y_coords


## DISTANCES // AREAS
def reggrid_dx(lats, lons):
    """returns a the cell widths (dx) for a regular grid with cell centers
    lats & lons [m]."""
    xres = np.abs(np.mean(np.diff(lons)))
    dx = degree_metres_x(lats) * xres
    return dx[:, None] * np.ones((lats.size, lons.size), dtype=lats.dtype)


def reggrid_dy(lats, lons):
    """returns a the cell heights (dy) for a regular grid with cell centers
    lats & lons [m]."""
    yres = np.abs(np.mean(np.diff(lats)))
    dy = degree_metres_y(lats) * yres
    return dy[:, None] * np.ones((lats.size, lons.size), dtype=lats.dtype)


def reggrid_area(lats, lons):
    """returns a the cell area for a regular grid with cell centres
    lats & lons [m2]."""
    xres = np.abs(np.mean(np.diff(lons)))
    yres = np.abs(np.mean(np.diff(lats)))
    area = np.ones((lats.size, lons.size), dtype=lats.dtype)
    return cellarea(lats, xres, yres)[:, None] * area


@njit
def cellarea(lat, xres, yres):
    """returns the area of cell with a given resolution (resx,resy) at a given
    cell center latitude [m2]."""
    l1 = np.radians(lat - np.abs(yres) / 2.0)
    l2 = np.radians(lat + np.abs(yres) / 2.0)
    dx = np.radians(np.abs(xres))
    return _R ** 2 * dx * (np.sin(l2) - np.sin(l1))


@njit
def degree_metres_y(lat):
    """ "returns the verical length of a degree in metres at
    a given latitude."""
    m1 = 111132.92  # latitude calculation term 1
    m2 = -559.82  # latitude calculation term 2
    m3 = 1.175  # latitude calculation term 3
    m4 = -0.0023  # latitude calculation term 4
    # # Calculate the length of a degree of latitude and longitude in meters
    radlat = np.radians(lat)
    latlen = (
        m1
        + (m2 * np.cos(2.0 * radlat))
        + (m3 * np.cos(4.0 * radlat))
        + (m4 * np.cos(6.0 * radlat))
    )
    return latlen


@njit
def degree_metres_x(lat):
    """ "returns the horizontal length of a degree in metres at
    a given latitude."""
    p1 = 111412.84  # longitude calculation term 1
    p2 = -93.5  # longitude calculation term 2
    p3 = 0.118  # longitude calculation term 3
    # # Calculate the length of a degree of latitude and longitude in meters
    radlat = np.radians(lat)
    longlen = (
        (p1 * np.cos(radlat))
        + (p2 * np.cos(3.0 * radlat))
        + (p3 * np.cos(5.0 * radlat))
    )
    return longlen


@njit
def distance(idx0, idx1, ncol, latlon=False, transform=IDENTITY):
    """Return the the length between linear indices idx0 and idx1 on a regular raster
    defined by the affine transform.

    Parameters
    ----------
    idx0, idx1 : int
        index of start, end cell
    ncol : int
        number of columns in raster
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping

    Returns
    -------
    float
        length
    """
    xres, yres, north = transform[0], transform[4], transform[5]
    # compute delta row, col
    r0 = idx0 // ncol
    r1 = idx1 // ncol
    dr = abs(r1 - r0)
    dc = abs((idx1 % ncol) - (idx0 % ncol))
    if latlon:  # calculate cell size in metres
        lat = north + (r0 + r1) / 2.0 * yres
        dy = 0.0 if dr == 0 else degree_metres_y(lat) * yres
        dx = 0.0 if dc == 0 else degree_metres_x(lat) * xres
    else:
        dy = xres
        dx = yres
    return math.hypot(dy * dr, dx * dc)  # length


## VECTORIZE
def features(flowpaths, xs, ys, **kwargs):
    """Returns a LineString feature for each stream

    Parameters
    ----------
    flowpaths : list of 1D-arrays of intp
        linear indices of flowpaths
    xs, ys : 1D-array of float
        x, y coordinates
    kwargs : extra sample maps key-word arguments
        optional maps to sample from
        e.g.: strord=flw.stream_order()

    Returns
    -------
    feats : list of dict
        Geofeatures, to be parsed by e.g. geopandas.GeoDataFrame.from_features
    """
    for key in kwargs:
        if not isinstance(kwargs[key], np.ndarray) or kwargs[key].size != xs.size:
            raise ValueError(
                f'Kwargs map "{key}" should be ndarrays of same size as coordinates'
            )
    feats = list()
    for idxs in flowpaths:
        if len(idxs) < 2:
            continue
        idx0 = idxs[0]
        props = {key: kwargs[key].flat[idx0] for key in kwargs}
        feats.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [(xs[i], ys[i]) for i in idxs],
                },
                "properties": {"idxs": idx0, "pit": idxs[-1] == idxs[-2], **props},
            }
        )
    return feats

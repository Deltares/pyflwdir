# -- coding: utf-8 --
""""""

from numba import njit
import numpy as np
import math
from affine import Affine
import heapq

_R = 6371e3  # Radius of earth in m. Use 3956e3 for miles
AREA_FACTORS = {"m2": 1.0, "ha": 1e4, "km2": 1e6, "cell": 1}
# changed to N->S orientation in v0.5 TODO check if used in hydromt?
IDENTITY = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

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
    "get_edge",
    "spread2d",
]


@njit()
def spread2d(obs, msk=None, nodata=0, frc=None, latlon=False, transform=IDENTITY):
    """Returns filled array with nearest observations, origin cells and friction distance to origin.
    The friction distance is measured through valid cells in the mask and has a uniform value of 1. by default.
    The diagonal distance is taken as the hypot of the vertical and horizontal distances.


    Parameters
    ----------
    osb: 2D array
        Initial array with observations.
    msk: 2D array of bool, optional
        Mask of valid cells to consider for filling.
    nodata: int, float
        Missing data value in obs. Cells with this value and where mask equals True are filled, by default 0.
    frc: 2D array of float
        Friction values, by default a uniform value of 1 is used.
    latlon: bool
        True for geographic CRS, False for projected CRS.
        If True, the transform units are assumed to be degrees and converted to metric distances.
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.

    Returns
    -------
    out: 2D array of obs.dtype
        Output observations array where nodata values are filled with the nearest observation.
    src: 2D array of int32
        Linear index of origin cell.
    dst: 2D array of float32
        Distance to origin cell.
    """
    nrow, ncol = obs.shape
    xres, yres, north = transform[0], abs(transform[4]), transform[5]
    if latlon:
        lats = north + (np.arange(nrow) + 0.5) * yres
        dys = degree_metres_y(lats) * yres
        dxs = degree_metres_x(lats) * xres
    else:
        dx, dy = xres, yres

    # output
    out = obs.copy()
    src = np.full(obs.shape, -1, dtype=np.int32)  # linear index of source
    dst = np.full(obs.shape, 0, dtype=np.float32)  # distance from source

    # initiate queue with correct dtype
    # heapq is faster when fifo loop not in order of ascending distance from source;
    # otherwise a fixed length numpy array queue is up to ~2x faster
    q = [(np.float32(0), np.uint32(0), np.uint32(0)) for _ in range(0)]
    heapq.heapify(q)

    for r in range(nrow):
        for c in range(ncol):
            if obs[r, c] != nodata:
                if msk is None or msk[r, c]:
                    heapq.heappush(q, (np.float32(0), np.uint32(r), np.uint32(c)))
                src[r, c] = r * ncol + c

    obs = obs.ravel()
    while len(q) > 0:
        d0, r, c = heapq.heappop(q)
        if dst[r, c] < d0:
            continue
        f0 = 1.0 if frc is None else frc[r, c]
        if latlon:
            dx, dy = dxs[r], dys[r]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                r1, c1 = int(r) + dr, int(c) + dc
                outside = r1 < 0 or r1 >= nrow or c1 < 0 or c1 >= ncol
                if outside or (msk is not None and ~msk[r1, c1]):
                    continue
                d = d0 + np.hypot(dr * dy, dc * dx) * f0
                if src[r1, c1] == -1 or d < dst[r1, c1]:
                    idx0 = src[r, c]
                    src[r1, c1] = idx0
                    dst[r1, c1] = d
                    out[r1, c1] = obs[idx0]
                    heapq.heappush(q, (np.float32(d), np.uint32(r1), np.uint32(c1)))

    return out, src, dst


@njit
def get_edge(a, structure=np.ones((3, 3), dtype=bool)):
    """Get edge of valid cells.

    Parameters
    ----------
    a: 2D array of bool
        Boolean array valid cells.
    structure: 2D array with shape (3,3) of bool
        Structuring element used to define which cells are neighbors.

    Returns
    -------
    edge: 2D array of bool
        Boolean array edge cells.
    """
    assert structure.shape == (3, 3)
    s = np.where(structure.ravel())[0]
    edge = a.copy()
    nrow, ncol = a.shape
    for r in range(0, nrow):
        for c in range(0, ncol):
            if ~a[r, c] or r == 0 or r == nrow - 1 or c == 0 or c == ncol - 1:
                continue
            a0 = a[slice(r - 1, r + 2), slice(c - 1, c + 2)].ravel()
            if np.all(a0[s]):
                edge[r, c] = False
    return edge


## TRANSFORM
# Adapted from https://github.com/rasterio/rasterio/blob/main/rasterio/transform.py
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
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
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
    transform: Affine
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
        eps = 10.0**-precision * (1.0 - 2.0 * op(0.1))
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
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
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
    idxs = np.asarray(idxs).astype(int)
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
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
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
    affine: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
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
    area = np.ones((lats.size, lons.size), dtype=np.float32)
    return cellarea(lats, xres, yres)[:, None] * area


def area_grid(transform, shape, latlon=False, unit="m2"):
    """Returns a regular grid with cell areas"""
    unit = str(unit).lower()
    if unit not in AREA_FACTORS:
        fstr = '", "'.join(AREA_FACTORS.keys())
        raise ValueError(f'Unknown unit: {unit}, select from "{fstr}".')
    if unit == "cell":
        area = np.ones(shape, dtype=np.int32)
    elif latlon:
        lon, lat = affine_to_coords(transform, shape)
        area = reggrid_area(lat, lon) / AREA_FACTORS[unit]
    elif not latlon:
        area0 = abs(transform[0] * transform[4]) / AREA_FACTORS[unit]
        area = np.full(shape, area0, dtype=np.float32)
    return area


@njit
def cellarea(lat, xres, yres):
    """returns the area of cell with a given resolution (resx,resy) at a given
    cell center latitude [m2]."""
    l1 = np.radians(lat - np.abs(yres) / 2.0)
    l2 = np.radians(lat + np.abs(yres) / 2.0)
    dx = np.radians(np.abs(xres))
    return _R**2 * dx * (np.sin(l2) - np.sin(l1))


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
    latlon: bool
        True for geographic CRS, False for projected CRS.
        If True, the transform units are assumed to be degrees and converted to metric distances.
    transform: Affine
        Coefficients mapping pixel coordinates to coordinate reference system.

    Returns
    -------
    float
        length
    """
    xres, yres, north = transform[0], transform[4], transform[5]
    # compute delta row, col
    r0 = int(idx0 // ncol)
    r1 = int(idx1 // ncol)
    dr = abs(r1 - r0)
    dc = abs(int(idx1 % ncol) - int(idx0 % ncol))
    if latlon:  # calculate cell size in metres
        lat = north + (r0 + r1) / 2.0 * yres
        dy = 0.0 if dr == 0 else degree_metres_y(lat) * yres
        dx = 0.0 if dc == 0 else degree_metres_x(lat) * xres
    else:
        dy = xres
        dx = yres
    return math.hypot(dy * dr, dx * dc)  # length


## VECTORIZE
def features(flowpaths, xs=None, ys=None, transform=None, shape=None, **kwargs):
    """Returns a LineString feature for each stream

    Parameters
    ----------
    flowpaths : list of 1D-arrays of intp
        linear indices of flowpaths
    xs, ys : 1D-array of float
        x, y coordinates
    transform : Affine
        Coefficients mapping pixel coordinates to coordinate reference system.
    shape : tuple of int
        The height, width  of the raster.
    kwargs : extra sample maps key-word arguments
        optional maps to sample from
        e.g.: strord=flw.stream_order()

    Returns
    -------
    feats : list of dict
        Geofeatures, to be parsed by e.g. geopandas.GeoDataFrame.from_features
    """
    if xs is None or ys is None:
        if transform is None or shape is None:
            raise ValueError(
                "transform and shape should be provided if xs and ys are None"
            )
        _size = shape[0] * shape[1]
    else:
        _size = xs.size

    for key in kwargs:
        if not isinstance(kwargs[key], np.ndarray) or kwargs[key].size != _size:
            raise ValueError(
                f'Kwargs map "{key}" should be ndarrays of same size as coordinates'
            )
    feats = list()
    for j, idxs in enumerate(flowpaths):
        n = len(idxs)
        if n < 2:
            continue
        idx0 = idxs[0]
        pit = idxs[-1] == idxs[-2]
        props = {key: kwargs[key].flat[idx0] for key in kwargs}
        if xs is None or ys is None:
            xi, yi = idxs_to_coords(idxs, transform, shape)
            coordinates = list(zip(xi, yi))
        else:
            coordinates = [(xs[i], ys[i]) for i in idxs]
        feats.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
                "properties": {"idx": idx0, "idx_ds": idxs[-1], "pit": pit, **props},
            }
        )
    return feats

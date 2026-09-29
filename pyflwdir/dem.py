"""Methods to derive topographic and hydrographic parameters from elevation data, in some cases
in combination with flow direction data."""

import heapq
import math
from typing import Literal

import numpy as np
from affine import Affine
from numba import njit

from . import core, core_d8, gis_utils

_mv = core._mv

__all__ = ["fill_depressions", "slope"]


@njit(cache=True)
def fill_depressions(
    elevtn: np.ndarray,
    outlets: Literal["edge", "min"] = "edge",
    idxs_pit: np.ndarray | None = None,
    nodata: float = -9999.0,
    max_depth: float = -1.0,
    elv_max: float | None = None,
    connectivity: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill local depressions in elevation data and derived local
    D8 flow directions.

    Outlets are assumed to occur at the edge of valid elevation cells `outlets='edge'`;
    at the lowest valid edge cell to create one single outlet `outlets='min'`;
    or at user provided outlet cells `idxs_pit`.

    Depressions elsewhere are filled to their lowest pour-point elevation. If the pour
    point depth is greater than or equal to `max_depth`, a pit is set at the depression's
    local minimum elevation.

    Based on: Wang, L., & Liu, H. (2006). https://doi.org/10.1080/13658810500433453

    Parameters
    ----------
    elevtn : 2D array
        elevation raster
    outlets : {'edge', 'min'}, optional
        Initialize outlets at valid edge cells ('edge', default) or use only the
        lowest-elevation valid edge cell ('min'). If `idxs_pit` is provided, `outlets`
        controls whether all supplied outlets or only the lowest-elevation one are used.
    idxs_pit : 1D array of int, optional
        Linear indices of user-specified outlet cells. By default, outlets are selected
        from the valid raster edge.
    nodata : float, optional
        No-data value, by default -9999.0.
    max_depth : float, optional
        Maximum pour point depth. Depressions with a larger pour point
        depth are set as pits. A negative value (default) represents an infinitely
        large pour point depth causing all depressions to be filled.
    elv_max : float, optional
        Maximum elevation for outlets, only used with `outlets='edge'`. By default None.
    connectivity : {4, 8}, optional
        Number of neighboring cells to consider.

    Returns
    -------
    elevtn_out : 2D array
        Depression-filled elevation raster.
    d8 : 2D array of uint8
        D8 flow directions, with no-data cells encoded as 247.
    """
    nrow, ncol = elevtn.shape
    delv = np.zeros_like(elevtn)
    done = np.isnan(elevtn) if np.isnan(nodata) else elevtn == nodata
    d8 = np.where(done, np.uint8(247), np.uint8(0))
    if connectivity not in [4, 8]:
        raise ValueError('"connectivity" should either be 4 or 8')
    # pfff.. numba does not allow creation of numpy bool arrays using normal methods
    struct = np.array([bool(1) for s in range(9)]).reshape((3, 3))
    if connectivity == 4:
        struct[0, 0], struct[-1, -1] = False, False
        struct[0, -1], struct[-1, 0] = False, False

    # initiate queue
    if idxs_pit is None:  # with edge cells
        queued = gis_utils._get_edge(~done, struct)
        if elv_max is not None:
            queued = np.logical_and(queued, elevtn <= elv_max)
            if not np.any(queued):
                raise ValueError("No initial outlet cells found.")
    else:  # with user defined outlet cells
        queued = np.array([bool(0) for s in range(elevtn.size)]).reshape((nrow, ncol))
        for idx in idxs_pit:
            queued.flat[idx] = True

    # queue contains (elevation, boundary, row, col)
    # boundary is included to favor non-boundary cells over boundary cells with same elevation
    q = [
        (np.float32(elevtn[0, 0]), np.uint8(1), np.uint32(0), np.uint32(0))
        for _ in range(0)
    ]
    heapq.heapify(q)
    for r, c in zip(*np.where(queued)):
        heapq.heappush(
            q, (np.float32(elevtn[r, c]), np.uint8(1), np.uint32(r), np.uint32(c))
        )
    # restrict queue to the global edge minimum (single outlet)
    if outlets == "min":
        q = [heapq.heappop(q)]
        queued[:, :] = False
        queued[q[0][-2], q[0][-1]] = True

    # loop over cells and neighbors with ascending cell elevation.
    drs, dcs = np.where(struct)
    drs, dcs = drs - 1, dcs - 1
    while len(q) > 0:
        z0, _, r0, c0 = heapq.heappop(q)
        for dr, dc in zip(drs, dcs):
            r = r0 + dr
            c = c0 + dc
            if r < 0 or r == nrow or c < 0 or c == ncol or done[r, c]:
                continue
            z1 = elevtn[r, c]
            dz = z0 - z1  # local depression if dz > 0
            if max_depth >= 0:  # if positive max_depth: don't fill when dz > max_depth
                if dz >= max_depth:
                    heapq.heappush(
                        q, (np.float32(z1), np.uint8(0), np.uint32(r), np.uint32(c))
                    )
                    queued[r, c] = True
                    for dr, dc in zip(drs, dcs):  # (re)visit neighbors
                        done[r + dr, c + dc] = False
                    continue
                elif delv[r, c] > 0:  # reset cell if previously filled & revisited
                    queued[r, c] = False
                    delv[r, c] = 0
            if dz > 0:  # check if local depression (dz>0)
                delv[r, c] = dz
                z1 += dz
            if ~queued[r, c]:  # add to queue
                heapq.heappush(
                    q, (np.float32(z1), np.uint8(0), np.uint32(r), np.uint32(c))
                )
                queued[r, c] = True
            done[r, c] = True
            d8[r, c] = core_d8._us[dr + 1, dc + 1]
    return elevtn + delv, d8


@njit(cache=True)
def adjust_elevation(
    idxs_ds: np.ndarray, seq: np.ndarray, elevtn: np.ndarray, mv: int = _mv
) -> np.ndarray:
    """Given a flow direction map, remove pits in the elevation map.
    Algorithm based on Yamazaki et al. (2012)

    Parameters
    ----------
    idxs_ds : 1D array of int
        Linear indices of the next downstream cell.
    seq : 1D array of int
        Valid cell indices ordered from downstream to upstream.
    elevtn : 1D array of float
        Flattened elevation raster.
    mv : int, optional
        Missing-index value, by default the package default.

    Returns
    -------
    1D array of float
        Adjusted flattened elevation values.

    .. ref: Yamazaki, D., Baugh, C. A., Bates, P. D., Kanae, S., Alsdorf, D. E. and
    Oki, T.: Adjustment of a spaceborne DEM for use in floodplain hydrodynamic
    modeling, J. Hydrol., 436-437, 81-91, doi:10.1016/j.jhydrol.2012.02.045,
    2012.
    """
    elevtn_out = elevtn.copy()
    mask = np.zeros(idxs_ds.size, dtype=np.bool_)
    for idx0 in seq[::-1]:  # from up- to downstream starting from longest stream paths
        if mask[idx0] == False:  # headwater cell
            # get downstream indices up to earlier fixed stream path
            idxs0 = core._trace(idx0, idxs_ds, mv=mv, mask=mask)[0]
            # fix elevation
            elevtn1 = _adjust_elevation(elevtn_out[idxs0])
            # assert np.all(np.diff(elevtn1) <= 0), elevtn_out[idxs0]
            elevtn_out[idxs0] = elevtn1
            mask[idxs0] = True  # update mask
    return elevtn_out


@njit(cache=True)
def _adjust_elevation(elevtn: np.ndarray) -> np.ndarray:
    """fix elevation on single streamline based on minimum modification
    elevtn ordered from upstream to downstream
    """
    n = elevtn.size
    imax, imin = -1, -1
    zmax, zmin = elevtn[0], elevtn[0]  # local max / min elevation
    zi_min1, zi_min2 = zmin, zmin  # initialize
    # all elevtn should be larger than last value
    elevtn = np.maximum(elevtn, elevtn[-1])
    for i in range(elevtn.size):
        zi = elevtn[i]
        if zi >= zmax:
            zmax = zi
            imax = i
        if (zi > zi_min1 and zi_min2 >= zi_min1) or (imin >= 0 and i + 1 == n):  # pit
            if imin >= 0:  # starting from second pit or end of vector
                # option 1: dig -> zmod = zmin, for all values larger than zmin, after imin
                idxs = np.arange(imin, i, dtype=np.uint32)
                zmod = np.minimum(zmin, elevtn[idxs])
                cost = np.sum(np.abs(elevtn[idxs] - zmod))
                # option 2: fill -> zmod = zmax, for all values smaller than zmax, previous to imax
                idxs2 = np.arange(0, imax, dtype=np.uint32)
                zmod2 = np.maximum(zmax, elevtn[idxs2])
                cost2 = np.sum(np.abs(elevtn[idxs2] - zmod2))
                if cost2 < cost:
                    cost, idxs, zmod = cost2, idxs2, zmod2
                # option 3: dig & fill -> try all values between imin and imax
                i0, j0, i1, j1 = 0, 0, imax, imax
                zs = np.unique(elevtn[imin + 1 : i])[::-1]
                for z in zs[1:]:  # skip zmax
                    for j0 in range(i0, imin + 1):  # start of zmod
                        if elevtn[j0] <= z:
                            break
                    for j1 in range(i1, i + 1):  # end of zmod
                        if elevtn[j1] <= z:
                            break
                    i0, i1 = j0, j1
                    idxs2 = np.arange(j0, max(imax + 1, j1), dtype=np.uint32)
                    zmod2 = np.full(idxs2.size, z, dtype=elevtn.dtype)
                    cost2 = np.sum(np.abs(elevtn[idxs2] - zmod2))
                    if cost2 < cost:
                        cost, idxs, zmod = cost2, idxs2, zmod2
                # update elevation
                elevtn[idxs] = zmod
            # update zmin & zmax
            imax = i
            zmax = elevtn[imax]
            imin = max(0, i - 1)
            zmin = elevtn[imin]
        # update zi values
        if zi_min2 != zi_min1:
            zi_min2 = zi_min1
        zi_min1 = zi
    return elevtn


@njit(cache=True)
def slope(
    elevtn: np.ndarray,
    nodata: float = -9999.0,
    latlon: bool = False,
    transform: np.ndarray = gis_utils._IDENTITY,
) -> np.ndarray:
    """Return the local slope magnitude.

    The slope is calculated from the DEM in a 3-by-3-cell window using second-order
    partial derivatives. It is the magnitude of the elevation gradient, in metres per
    metre.

    Parameters
    ----------
    elevtn : 2D array of float
        Elevation raster.
    nodata : float, optional
        No-data value, by default -9999.0.
    latlon : bool, optional
        True if coordinates use the WGS84 geographic coordinate system, by default False.
    transform : np.ndarray, optional
        2D array with 6 elements representing the affine transformation for raster,
        By default, the identity transform `(1, 0, 0, 0, -1, 0)`.

    Returns
    -------
    2D array of float
        Slope magnitude [m/m].
    """
    xres, yres, north = transform[0], transform[4], transform[5]
    slope = np.zeros(elevtn.shape, dtype=np.float32)
    nrow, ncol = elevtn.shape

    elev = np.zeros((3, 3), dtype=elevtn.dtype)

    for r in range(nrow):
        for c in range(ncol):
            if elevtn[r, c] != nodata:
                # start with matrix based on central value (inside loop)
                elev[:, :] = elevtn[r, c]

                for dr in range(-1, 2):
                    row = r + dr
                    i = dr + 1
                    if row >= 0 and row < nrow:
                        for dc in range(-1, 2):
                            col = c + dc
                            j = dc + 1
                            # fill matrix with elevation, except when nodata
                            if 0 <= col < ncol and elevtn[row, col] != nodata:
                                elev[i, j] = elevtn[row, col]

                dzdx = (
                    (elev[0, 0] + 2 * elev[1, 0] + elev[2, 0])
                    - (elev[0, 2] + 2 * elev[1, 2] + elev[2, 2])
                ) / (8 * abs(xres))
                dzdy = (
                    (elev[0, 0] + 2 * elev[0, 1] + elev[0, 2])
                    - (elev[2, 0] + 2 * elev[2, 1] + elev[2, 2])
                ) / (8 * abs(yres))

                if latlon:
                    lat = north + (r + 0.5) * yres
                    deg_y = gis_utils.degree_metres_y(lat)
                    deg_x = gis_utils.degree_metres_x(lat)
                    slp = math.hypot(dzdx / deg_x, dzdy / deg_y)
                else:
                    slp = math.hypot(dzdx, dzdy)
            else:
                slp = nodata

            slope[r, c] = slp

    return slope


def height_above_nearest_drain(
    idxs_ds: np.ndarray, seq: np.ndarray, drain: np.ndarray, elevtn: np.ndarray
) -> np.ndarray:
    """Returns the height above the nearest drain (HAND), i.e.: the relative vertical
    distance (drop) to the nearest downstream river based on drainage-normalized
    topography and flowpaths.

    Nobre A D et al. (2016) HAND contour: a new proxy predictor of inundation extent
        Hydrol. Process. 30 320–33

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    drain : 1D array of bool
        flattened drainage mask
    elevtn : 1D array of float
        Flattened elevation raster.

    Returns
    -------
    1D array of float
        height above nearest drain
    """
    hand = np.full(drain.size, -9999.0, dtype=np.float64)
    hand[seq] = 0.0
    for idx0 in seq:
        if drain[idx0] != 1:
            idx_ds = idxs_ds[idx0]
            dz = elevtn[idx0] - elevtn[idx_ds]
            hand[idx0] = hand[idx_ds] + dz
    return hand


def floodplains(
    idxs_ds: np.ndarray,
    seq: np.ndarray,
    elevtn: np.ndarray,
    uparea: np.ndarray,
    upa_min: float = 1000.0,
    b: float = 0.3,
) -> np.ndarray:
    """Identify floodplain cells using an upstream-area-scaled HAND threshold.

    Cells with upstream area at least `upa_min` define the drainage network. For each
    such cell, the HAND threshold is its upstream area raised to `b`; upstream cells
    are included when their elevation above the downstream drainage cell does not
    exceed that threshold.

    Nardi F et al (2019) GFPLAIN250m, a global high-resolution dataset of Earth's
        floodplains Sci. Data 6 180309

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    elevtn : 1D array of float
        Flattened elevation raster [m].
    uparea : 1D array of float
        flattened upstream area raster [km2]
    upa_min : float, optional
        Minimum upstream-area threshold for drainage cells [km2], by default 1000.
    b : float
        Exponent in the upstream-area scaling relationship, by default 0.3.

    Returns
    -------
    1D array of int8
        Floodplain mask: 1 for floodplain cells, 0 for valid non-floodplain cells, and
        -1 for no-data cells.
    """
    drainh = np.full(uparea.size, -9999.0, dtype=np.float32)
    drainz = np.full(uparea.size, -9999.0, dtype=np.float32)
    fldpln = np.full(uparea.size, -1, dtype=np.int8)
    fldpln[seq] = 0
    for idx0 in seq:  # down- to upstream
        if uparea[idx0] >= upa_min:
            drainh[idx0] = uparea[idx0] ** b
            drainz[idx0] = elevtn[idx0]
            fldpln[idx0] = 1
        else:
            idx_ds = idxs_ds[idx0]
            if fldpln[idx_ds] == 1:
                z0 = drainz[idx_ds]
                h0 = drainh[idx_ds]
                dh = elevtn[idx0] - z0
                if dh <= h0:
                    fldpln[idx0] = 1
                    drainz[idx0] = z0
                    drainh[idx0] = h0
    return fldpln


@njit(cache=True)
def _local_d4(idx0: int, idx_ds: int, ncol: int) -> np.ndarray:
    """Return D4 neighbors for a diagonal D8 flow direction.

    For example, a northwest flow direction returns the north and west neighbors.
    """
    idxs_d4 = [
        idx0 - ncol,
        idx0 - 1,
        idx0 + ncol,
        idx0 + 1,
        idx0 - ncol,
    ]  # n, w, s, e, n
    if idx_ds != idx0:
        idxs_diag = [
            idx0 - ncol - 1,
            idx0 + ncol - 1,
            idx0 + ncol + 1,
            idx0 - ncol + 1,
        ]  # nw, sw, se, ne
        di = idxs_diag.index(idx_ds)
        return np.asarray(idxs_d4[di : di + 2])
    else:
        return np.asarray(idxs_d4[1:])


@njit(cache=True)
def dig_4connectivity(
    idxs_ds: np.ndarray,
    seq: np.ndarray,
    elv_flat: np.ndarray,
    shape: tuple[int, int],
    mask: np.ndarray | None = None,
    nodata: float = -9999,
    dz_min: float = 1e-3,
) -> np.ndarray:
    """Make sure that for every diagonal D8 downstream flow direction
    there is an adjacent D4 cell with same or lower elevation"""
    elv_out = elv_flat.copy()
    nrow, ncol = shape
    for idx0 in seq[::-1]:  # up- to downstream
        if mask is not None and not mask[idx0]:
            continue
        idx_ds = idxs_ds[idx0]
        dd = abs(idx0 - idx_ds)
        if dd > 1 and dd != ncol:  # diagonal
            idxs_d4 = _local_d4(idx0, idx_ds, ncol)  # indices of adjacent d4 cells
            z0 = elv_out[idx0]  # elevtn of current cell
            zs = elv_out[idxs_d4]
            valid = zs != nodata
            if not np.any(valid):
                continue
            # find adjacent with smallest dz and lower elevation to <= z0
            idx_d4_min = idxs_d4[valid][np.argmin(zs[valid] - z0)]
            # force small change to detect d4 river
            elv_out[idx_d4_min] = min(elv_out[idx_d4_min] - dz_min, z0)
        if idxs_ds[idx_ds] == idx_ds:  # next pit because we need to know upstream cell
            r = idx_ds // ncol
            c = idx_ds % ncol
            if r == 0 or r == nrow - 1 or c == 0 or c == ncol - 1:  # edge
                continue
            idxs_d4 = _local_d4(idx_ds, idx_ds, ncol)
            if np.any(elv_out[idxs_d4] == nodata):  # D4 link with nodata
                continue
            idxs_d4 = np.asarray([idx for idx in idxs_d4 if idx != idx0])
            elv_out[idxs_d4] = np.minimum(elv_out[idx_ds], elv_out[idxs_d4])
    return elv_out

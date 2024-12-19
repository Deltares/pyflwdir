# -*- coding: utf-8 -*-
"""Methods to derive topo/hydrographical paramters from elevation data, in some cases
 in combination with flow direction data."""

import numpy as np
from numba import njit
import math
import heapq

from . import gis_utils, core, core_d8

_mv = core._mv

__all__ = ["slope", "fill_depressions"]


@njit
def fill_depressions(
    elevtn,
    outlets="edge",
    idxs_pit=None,
    nodata=-9999.0,
    max_depth=-1.0,
    elv_max=None,
    connectivity=8,
):
    """Fill local depressions in elevation data and derived local
    D8 flow directions.

    Outlets are assumed to occur at the edge of valid elevation cells `outlets='edge'`;
    at the lowest valid edge cell to create one single outlet `outlets='min'`;
    or at user provided outlet cells `idxs_pit`.

    Depressions elsewhere are filled based on its lowest pour point elevation.
    If the pour point depth is larger than the maximum pour point depth `max_depth` a pit
    is set at the depression local minimum elevation.

    Based on: Wang, L., & Liu, H. (2006). https://doi.org/10.1080/13658810500433453

    Parameters
    ----------
    elevtn: 2D array
        elevation raster
    nodata: float, optional
        nodata value, by default -9999.0
    max_depth: float, optional
        Maximum pour point depth. Depressions with a larger pour point
        depth are set as pit. A negative value (default) equals an infitely
        large pour point depth causing all depressions to be filled.
    connectivity: {4, 8}
        Number of neighboring cells to consider.
    outlets: {'edge', 'min'}
        Initial basin outlet(s) at the edge of all cells ('edge'; default)
        or only the minimum elevation edge cell ('min')
    elv_max, float, optional
        Maximum elevation for outlets, only in combination with `outlets='edge'`.
        By default None.
    idxs_pit: 1D array of int
        Linear indices of outlet cells.

    Returns
    -------
    elevtn_out: 2D array
        Depression filled elevation
    d8: 2D array of uint8
        D8 flow directions
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
        queued = gis_utils.get_edge(~done, structure=struct)
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
    # restrict queue to global edge mimimum (single outlet)
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


@njit
def adjust_elevation(idxs_ds, seq, elevtn, mv=_mv):
    """Given a flow direction map, remove pits in the elevation map.
    Algorithm based on Yamazaki et al. (2012)

    .. ref: Yamazaki, D., Baugh, C. A., Bates, P. D., Kanae, S., Alsdorf, D. E. and
    Oki, T.: Adjustment of a spaceborne DEM for use in floodplain hydrodynamic
    modeling, J. Hydrol., 436-437, 81-91, doi:10.1016/j.jhydrol.2012.02.045,
    2012.
    """
    elevtn_out = elevtn.copy()
    mask = np.array([bool(0) for _ in range(elevtn.size)])  # True for checked cells
    for idx0 in seq[::-1]:  # from up- to downstream starting from longest stream paths
        if mask[idx0] == False:  # @ head water cell
            # get downstream indices up to earlier fixed stream path
            idxs0 = core._trace(idx0, idxs_ds, mv=mv, mask=mask)[0]
            # fix elevation
            elevtn1 = _adjust_elevation(elevtn_out[idxs0])
            # assert np.all(np.diff(elevtn1) <= 0), elevtn_out[idxs0]
            elevtn_out[idxs0] = elevtn1
            mask[idxs0] = True  # update mask
    return elevtn_out


@njit
def _adjust_elevation(elevtn):
    """fix elevation on single streamline based on minimum modification
    elevtn oderdered from up- to downstream
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


@njit
def slope(elevtn, nodata=-9999.0, latlon=False, transform=gis_utils.IDENTITY):
    """Returns the local gradient

    The slope is calculated on the basis of the dem in a 3 x 3 cell window, using 2nd order partial derivatives.
    The slope [m/m] is given as the increase in height per distance in horizontal direction.

    Parameters
    ----------
    elevnt : 1D array of float
        elevation raster
    nodata : float, optional
        nodata value, by default -9999.0
    latlon : bool, optional
        True if WGS84 coordinates, by default False
    transform : affine transform
        Two dimensional transform for 2D linear mapping, by default gis_utils.IDENTITY

    Returns
    -------
    1D array of float
        slope [m/m]
    """
    xres, yres, north = transform[0], transform[4], transform[5]
    slope = np.zeros(elevtn.shape, dtype=np.float32)
    nrow, ncol = elevtn.shape

    elev = np.zeros((3, 3), dtype=elevtn.dtype)

    for r in range(0, nrow):
        for c in range(0, ncol):
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
                            if col >= 0 and col < ncol:
                                # fill matrix with elevation, except when nodata
                                if elevtn[row, col] != nodata:
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


def height_above_nearest_drain(idxs_ds, seq, drain, elevtn):
    """Returns the height above the nearest drain (HAND), i.e.: the relative vertical
    distance (drop) to the nearest dowstream river based on drainage‐normalized
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
    elevnt : 1D array of float
        flattened elevation raster

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


def floodplains(idxs_ds, seq, elevtn, uparea, upa_min=1000.0, b=0.3):
    """Returns floodplain boundaries based on a maximum treshold (h) of HAND which is
    scaled with upstream area following h ~ A**b.

    Nardi F et al (2019) GFPLAIN250m, a global high-resolution dataset of Earth’s
        floodplains Sci. Data 6 180309

    Parameters
    ----------
    idxs_ds : 1D-array of intp
        index of next downstream cell
    seq : 1D array of int
        ordered cell indices from down- to upstream
    elevnt : 1D array of float
        flattened elevation raster [m]
    uparea : 1D array of float
        flattened upstream area raster [km2]
    upa_min : float, optional
        minimum upstream area threshold for streams.
    b : float
        scale parameter

    Returns
    -------
    1D array of int8
        floodplain
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


@njit
def _local_d4(idx0, idx_ds, ncol):
    """Return indices of d4 neighbors in diagonal d8 direction, e.g.: indices of N, W neigbors if flowdir is NW."""
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


@njit
def dig_4connectivity(
    idxs_ds, seq, elv_flat, shape, mask=None, nodata=-9999, dz_min=1e-3
):
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

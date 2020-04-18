# -*- coding: utf-8 -*-
"""Methods to derive topo/hydrographical paramters from elevation data, in some cases 
 in combination with flow direction data."""

import numpy as np
from numba import njit
import math

from pyflwdir import gis_utils, core
from pyflwdir.core import _mv

__all__ = ["slope"]


# TODO
def from_dem():
    """derive flow direction from elevation data"""
    raise NotImplementedError()


@njit
def adjust_elevation(idxs_ds, seq, elevtn):
    """Given a flow direction map, remove pits in the elevation map.
    Algorithm based on Yamazaki et al. (2012)
    
    Yamazaki, D., Baugh, C. A., Bates, P. D., Kanae, S., Alsdorf, D. E. and 
    Oki, T.: Adjustment of a spaceborne DEM for use in floodplain hydrodynamic 
    modeling, J. Hydrol., 436-437, 81-91, doi:10.1016/j.jhydrol.2012.02.045,
    2012.
    """
    elevtn_out = elevtn.copy()
    n_up = core.upstream_count(idxs_ds)
    for idx0 in seq[::-1]:  # from up- to downstream
        if n_up[idx0] <= 0:
            # @ head water cell, i.e. no upstream neighbors
            # get downstream indices
            idxs0 = core._trace(idx0, idxs_ds)[0]
            # fix elevation
            elevtn_out[idxs0] = _adjust_elevation(elevtn_out[idxs0])
    return elevtn_out


@njit
def _adjust_elevation(elevtn):
    """fix elevation on single streamline based on minimum modification
    elevtn oderdered from up- to downstream
    """
    n = elevtn.size
    zmin = elevtn[0]
    zmax = elevtn[0]
    valid = True
    for i in range(elevtn.size):
        zi = elevtn[i]
        if valid:
            if zi <= zmin:  # sloping down. do nothing
                zmin = zi
            else:  # new pit: downstream z > upstream z
                valid = False
                zmax = zi
                imax = i
                imin = i - 1
        if not valid:
            if zi <= zmin or i + 1 == elevtn.size:  # end of pit area: FIX
                # option 1: dig -> zmod = zmin, for all values after pit
                idxs = np.arange(imin, min(n, i + 1))
                zmod = np.full(idxs.size, zmin, dtype=elevtn.dtype)
                cost = np.sum(elevtn[idxs] - zmod)
                if (imax - imin) > 1:  # all options are equal when imax = imin + 1
                    # option 2: fill -> zmod = zmax, for all values smaller than zmax, previous to zmax
                    idxs2 = np.where(elevtn[:imax] <= zmax)[0]
                    zmod2 = np.full(idxs2.size, zmax, dtype=elevtn.dtype)
                    cost2 = np.sum(zmod2 - elevtn[idxs2])
                    if cost2 < cost:
                        cost, idxs, zmod = cost2, idxs2, zmod2
                    # option 3: dig and fill -> zmin < zmod < zmax
                    idxs3 = np.where(
                        np.logical_and(elevtn[:i] >= zmin, elevtn[:i] <= zmax)
                    )[0]
                    zorg = elevtn[idxs3]
                    for z3 in np.unique(zorg):
                        if z3 > zmin and z3 < zmax:
                            zmod3 = np.full(idxs3.size, z3, dtype=elevtn.dtype)
                            i0 = 0
                            i1 = zorg.size - 1
                            while zorg[i0] > z3:  # elevtn > z3 from start can remain
                                zmod3[i0] = zorg[i0]
                                i0 += 1
                            while zorg[i1] < z3:  # elevtn < z3 from end can remain
                                zmod3[i1] = zorg[i1]
                                i1 -= 1
                            cost3 = np.sum(np.abs(zmod3 - elevtn[idxs3]))
                            if cost3 < cost:
                                cost, idxs, zmod = cost3, idxs3, zmod3
                # adjust elevtn
                elevtn[idxs] = zmod
                zmin = zi
                valid = True
            elif zi >= zmax:  # between zmin and zmax (slope up) # get last imax (!)
                zmax = zi
                imax = i
    return elevtn


@njit
def slope(elevtn, nodata=-9999.0, latlon=False, transform=gis_utils.IDENTITY):
    """

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
        slop
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
    for idx0 in seq:
        if drain[idx0] == 1:
            hand[idx0] = 0
        else:
            idx_ds = idxs_ds[idx0]
            dz = elevtn[idx0] - elevtn[idx_ds]
            hand[idx0] = hand[idx_ds] + dz
    return hand


def floodplains(idxs_ds, seq, drain, elevtn, uparea, b=0.3):
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
    drain : 1D array of bool
        flattened drainage mask
    elevnt : 1D array of float
        flattened elevation raster [m]
    uparea : 1D array of float
        flattened upstream area raster [m2]
    b : float
        scale parameter

    Returns
    -------
    1D array of int8
        floodplain 
    """
    drainh = np.full(drain.size, -9999.0, dtype=np.float32)
    drainz = np.full(drain.size, -9999.0, dtype=np.float32)
    fldpln = np.full(drain.size, -1, dtype=np.int8)
    fldpln[seq] = 0
    for idx0 in seq:  # down- to upstream
        if drain[idx0] == 1:
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

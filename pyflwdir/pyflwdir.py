# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
#
# This script provides a python object with functions that basically wrap all numba functions
# in accompaning scripts

import logging
import numpy as np
from numba import njit
import rasterio
from rasterio.transform import Affine, array_bounds
from copy import deepcopy

# local
from .core import fd
from . import flux, utils, features, network, gridtools, d8_scaling, catchment, dem
# export
__all__ = ['FlwdirRaster']

# 
logger = logging.getLogger(__name__)

# global variabels
IDENTITY_NS = Affine(1, 0, 0, 0, -1, 0)

class FlwdirRaster(object):

    def __init__(self, data, transform=IDENTITY_NS, crs=4326, check_format=False, copy=False):
        """Create an instance of a pyflwdir.FlwDirRaster object
        
        Arguments:
            data {np.ndarray(unint8)} -- 2D flow direction array in D8 format
        
        Keyword Arguments:
            transform {Fffine.Transform} -- The GeoTransform of the flow direction map (default: {IDENTITY_NS})
            crs {int} -- The coordinate reference system if the flow direciton map, can be epsg code (default: {4326})
            check_format {bool} -- If True: check if the flow direction map is of D8 format (default: {False})
            copy {bool} -- If True: create a copy of the flow direciton map (default: {False})
        
        Raises:
            ValueError: If the data is too large for uint32 indices or incorrect 
        """
        assert data.ndim == 2
        assert np.sign(transform[4]) == -1 # North to South orientation
        self.d8format = fd._format
        # data
        self.shape = data.shape
        self.size = data.size
        if self.size > 2**32-2:
            raise ValueError('Extent too large for uint32 network indices')
        if copy:
            self._data = data.copy()
        else:
            self._data = data.view() # view of data
        self._data_flat = self._data.ravel() # flattened view of data
        if check_format and fd._check_format(self._data) == False: # simple check. isvalid includes more rigorous check
            raise ValueError('Unknown flow direction values found in data')
        # spatial meta data
        self.crs = rasterio.crs.CRS.from_user_input(crs)
        self.latlon = self.crs.is_geographic # NOTE: we assume this is in latlon
        self.transform = transform
        self.res = transform[0], transform[4]
        self.cellare = np.abs(transform[0]*transform[4])
        self.bounds = array_bounds(data.shape[0], data.shape[1], transform)
        # set placeholder properties for network
        self._idx0 = None            # most downstream indices in network
        self._rnodes = None          # network ds nodes (n)
        self._rnodes_up = None       # network us nodes (n,m); m <= 8 in d8

    def __getitem__(self, key):
        return self._data_flat[key]

    def __setitem__(self, key, item):
        self._data_flat[key] = item

    def _xycoords(self):
        resx, resy = self.res
        xmin, ymin, xmax, ymax = self.bounds
        if resy < 0:
            ys = np.linspace(ymax+resy/2., ymin-resy/2., self.shape[0])
        else:
            ys = np.linspace(ymin+resy/2., ymax-resy/2., self.shape[0])
        xs = np.linspace(xmin+resx/2., xmax-resx/2., self.shape[1])
        return xs, ys

    def isvalid(self):
        """Returns True if the flow direction map is valid, meaning that all cells flow to a pit/outlet.
        
        Returns:
            {boolean} -- valid
        """
        # check if all cells connect to pit / outflows at bounds
        valid = utils.flwdir_check(self._data_flat, self.shape)[1] == False
        return valid

    def repair(self):
        """Repair the flow direction map in order to have all cells flow to a pit."""
        repair_idx, _ = utils.flwdir_check(self._data_flat, self.shape)
        if repair_idx.size > 0:
            self._data_flat[repair_idx] = fd._pits[-1] # set inland pit

    def setup_network(self, idx0=None):
        """Setup all upstream - downstream connections based on the flow direcion map.
        
        Keyword Arguments:
            idx0 {np.ndayy(int)} -- Array with 1D outlet indices. 
                If none all pits in the flow direction map are used (default: {None})
        
        Raises:
            ValueError: Outlet indices are outside the map domain
        """
        if idx0 is None:
            idx0 = self.get_pits()
        elif not np.all(np.logical_and(idx0>=0, idx0<self.size)):
            raise ValueError('Outlet indices are outside the map domain')
        self._idx0 = np.atleast_1d(np.asarray(idx0, dtype=np.uint32)) # basin outlets
        self._rnodes, self._rnodes_up = network.setup_dd(self._idx0, self._data_flat, self.shape)

    def get_pits(self):
        """Return the indices of the pits/outlets in the flow direction map.
        
        Raises:
            ValueError: The flow direction data is not valid: no pits/outlets are found.
        
        Returns:
            np.ndarray(int) -- Indices of pits/outlets
        """
        idx0 = fd.pit_indices(self._data.flatten())
        if idx0.size == 0:
            raise ValueError('The flow direction data is not valid: no pits/outlets are found.')   
        return idx0 

    def upstream_area(self):
        """Returns the upstream area [km] based on the flow direction map. The cell area is 
        converted to metres is the map's coordinate reference system is geopgraphic.
        
        Returns:
            np.ndarray(float) -- Upstream area map
        """
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        if self.latlon:
            _, ys = self._xycoords()
            cellare = gridtools.lat_to_area(ys)*self.cellare/1e6
        else:
            cellare = np.ones(self.shape[0])*self.cellare/1e6
        return network.upstream_area(self._rnodes, self._rnodes_up, cellare, self.shape)

    def delineate_basins(self, idx=None):
        """Returns a map with basin ids and corresponding bounding boxes. This function does not
        use the up- downstream network but is directly derived from the flow direction map.
        
        Keyword Arguments:
            idx {np.ndarray(int)} -- List with 1D outlet indices (default: {None})
        
        Raises:
            ValueError: [description]
        
        Returns:
            (np.ndarray(int), np.ndarray(float)) -- Basin map and 2D bboxs (y: basins, x: (w, s, e, n))
        """
        if idx is None:
            idx = self.get_pits()
        elif not np.all(np.logical_and(idx>=0, idx<self.size)):
            raise ValueError('idx0 indices outside domain')
        idx = np.atleast_1d(np.asarray(idx, dtype=np.uint32)) # basin outlets
        resx, resy = self.res
        xs, ys = self._xycoords()
        return catchment.delineate_basins(idx, self._data_flat, self.shape, ys, xs, resy, resx)

    def basin_map(self, idx=None, values=None, dtype=np.int32):
        """Return a map with (sub)basins based on the up- downstream network.
        
        Keyword Arguments:
            idx {np.ndarray(int)} -- List/Array with 1D outlet indices (default: {None})
            values {np.ndarray(int)} -- List/Array with basin ids, should all be larger than zero (default: {None})
            dtype {np.dtype} -- numpy datatype for output map (default: {np.int32})
        
        Raises:
            ValueError: All values should be larger than zero
            ValueError: Idx and values should be 1d arrays of same size
        
        Returns:
            np.ndarray(float) -- (Sub)Basin map
        """
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        if idx is None:             # use most downstream network indices if idx not given
            idx = self._idx0
        idx = np.atleast_1d(idx)
        if values is None:          # number basins using range, starting from 1
            values = np.arange(idx.size, dtype=dtype)+1
        else:
            values = np.atleast_1d(values).astype(dtype)
            if np.any(values<=0):
                raise ValueError('All values should be larger than zero')
        if not idx.size == values.size and idx.ndim == 1:
            raise ValueError('Idx and values should be 1d arrays of same size')
        return network.basin_map(self._rnodes, self._rnodes_up, idx, values, self.shape)

    def ucat_map(self, scale_ratio, uparea=None, upa_min=0.5):
        """Returns the unit-subcatchment and outlets map, based on outlets, i.e. most downstream cells
        of low resolution regular gridcells. The scale ratio determines the size of the regular grid
        relative to the flow direction map resoltuion.
        
        Arguments:
            scale_ratio {int} -- The upscaling ratio
        
        Keyword Arguments:
            uparea {np.ndarray(float)} -- Upstream area map (default: {None})
            upa_min {float} -- Minimum upstream area for rivers (default: {0.5})
        
        Raises:
            ValueError: The shape of the uparea map does not match the flow direction data.
        
        Returns:
            (np.ndarray(int), np.ndarray(int))  -- Unitcatchment subbasin, outlet map
        """
        if uparea is None:
            uparea = self.upstream_area()
        elif not np.all(uparea.shape == self.shape):
            raise ValueError("The shape of the uparea map does not match the flow direction data.")
        # get unit catchment outlets based on d8_scaling
        uparea_flat = uparea.ravel()
        idx = d8_scaling.d8_scaling(
            scale_ratio, self._data_flat, uparea_flat, self.shape, upa_min=upa_min, extended=False
        )[1].ravel()
        upa_idx = uparea_flat[idx]
        idx = idx[upa_idx >= 0]
        upa_idx = upa_idx[upa_idx >= 0]
        values = (np.argsort(upa_idx)[::-1]+1).astype(np.int32) # number basins from large to small
        basins = self.basin_map(idx=idx, values=values, dtype=np.int32)
        outlet = np.zeros(self.shape, dtype=np.int32)
        outlet.flat[idx] = values
        return basins, outlet

    def basin_shape(self, basin_map=None, nodata=0, **kwargs):
        """Returns the vectorized basin boundary. In case no basin_map is given, it is calculated based on
        the outlet/pits in the domain.
        
        Keyword Arguments:
            basin_map {np.ndarray(int)} -- Basin map (default: {None})
            nodata {int} -- The nodata value of the Basin map (default: {0})
        
        Returns:
            geopandas.GeoDataFrame -- vectorized basin boundary
        """
        if basin_map is None:
            basin_map = self.basin_map(**kwargs)
            nodata = 0 # overwrite nodata
        return gridtools.vectorize(basin_map, nodata, self.transform, crs=self.crs)

    def stream_order(self):
        """Returns the Strahler Order map (TODO ref). 
        The smallest streams, which are the cells with no upstream cells, get an order 1. 
        Where two channels of order 1 join, a channel of order 2 results downstream. 
        In general, where two channels of order i join, a channel of order i+1 results.
        
        Returns:
            np.ndarray(int) -- Strahler Order map
        """
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        return network.stream_order(self._rnodes, self._rnodes_up, self.shape)

    def stream_shape(self, stream_order=None, mask=None, min_order=3, xs=None, ys=None):
        """Returns a GeoDataFrame with vectorized river segments. Segments are LineStrings 
        connecting cell from down to upstream and are splitted based on strahler order. 
        The coordinates of the nodes are based on the cell center, unless maps with x and y 
        coordinates are given.
        
        Keyword Arguments:
            stream_order {np.ndarray(int)} -- Strahler Order map (default: {None})
            mask {np.ndarray(bool)} -- Mask of valid area (default: {None})
            min_order {int} -- Minimum Strahler Order recognized as river (default: {3})
            xs {np.ndarray(float)} -- Map with cell x coordinates (default: {None})
            ys {np.ndarray(float)} -- Map with cell y coordinates (default: {None})
        
        Returns:
            geopandas.GeoDataFrame -- vectorized river segments
        """
        if stream_order is None:
            stream_order = self.stream_order()
        if mask is not None:
            stream_order[~mask] = -1
        else:
            stream_order[stream_order < min_order] = -1
        idx0 = self.get_pits() if self._idx0 is None else self._idx0
        # get lists of river nodes per stream order
        riv_nodes, riv_order = features.river_nodes(idx0, self._data_flat, stream_order.reshape(-1), self.shape)
        if xs is None or ys is None:
            xs, ys = self._xycoords()
        # create geometries
        pit_pnts = gridtools.nodes_to_pnts(idx0, ys, xs)
        riv_ls = [gridtools.nodes_to_ls(nodes, ys, xs) for nodes in riv_nodes]
        # make geopandas dataframe
        gdf_riv = gridtools.gp.GeoDataFrame(data=riv_order, columns=['stream_order'], geometry=riv_ls, crs=self.crs)
        gdf_pit = gridtools.gp.GeoDataFrame(data=idx0, columns=['idx'], geometry=pit_pnts, crs=self.crs)
        return gdf_riv, gdf_pit


    def upscale(self, scale_ratio, uparea=None, upa_min=0.5, method='extended'):
        """Returns upscaled flow direction map using the extended effective area method (Eilander et al., 2019) 
        
        Arguments:
            scale_ratio {int} -- The upscaling ratio
        
        Keyword Arguments:
            uparea {np.ndarray(float)} -- Upstream area map (default: {None})
            upa_min {float} -- Minimum upstream area for rivers (default: {0.5})
            method {str} -- upscaling method (default: {'extended'})
        
        Raises:
            ValueError: The data shape should be an exact multiplicity of the scale_ratio
        
        Returns:
            (pyflwdir.FlwDirRaster, np.ndarray(int)) -- Upscaled flow direction map, outlet map
        """
        if not self.shape[0] % scale_ratio == self.shape[1] % scale_ratio == 0:
            raise ValueError(f'The data shape should be an exact multiplicity of the scale_ratio')
        if uparea is None:
            uparea = self.upstream_area()
        transform_lr = Affine(
            self.transform[0] * scale_ratio, self.transform[1], self.transform[2],
            self.transform[3], self.transform[4] * scale_ratio, self.transform[5]
        )
        extended=method=='extended'
        uparea_flat = uparea.ravel()
        flwdir_lr, outlet_lr, changd_lr = d8_scaling.d8_scaling(
            scale_ratio=scale_ratio, flwdir_flat=self._data_flat, uparea_flat=uparea_flat, 
            shape=self.shape, upa_min=upa_min, extended=extended
        )
        flwdir_lr = FlwdirRaster(flwdir_lr, transform=transform_lr, crs=self.crs)
        return flwdir_lr, outlet_lr, changd_lr

    def propagate_downstream(self, material):
        """Returns a map with accumulated material from all upstream cells that 
        flow into the neighboring downstream cell based on the flow direction map. 
        
        Arguments:
            material {np.ndatray} -- Map with any material ammounts (i.e. state) to propagate
        
        Raises:
            ValueError: The shape of the material and flow direction map do not match
        
        Returns:
            np.ndarray -- Accumulated material map
        """
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        if not np.all(self.shape == material.shape):
            raise ValueError(f'The shape of material and flow direction map do not match.')
        return flux.propagate_downstream(self._rnodes, self._rnodes_up, material.copy().ravel(), self.shape)

    def propagate_upstream(self, material):
        """Returns a map with accumulated material from all downstream cells that 
        flow into the neighboring upstream cell based on the flow direction map. 
        
        Arguments:
            material {np.ndatray} -- Map with any material ammounts (i.e. state) to propagate
        
        Raises:
            ValueError: The shape of the material and flow direction map do not match
        
        Returns:
            np.ndarray -- Accumulated material map
        """
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        if not np.all(self.shape == material.shape):
            raise ValueError(f'The shape of the material and flow direction map do not match.')
        return flux.propagate_upstream(self._rnodes, self._rnodes_up, material.copy().ravel(), self.shape)

    def adjust_elevation(self, elevtn, copy=True):
        """Returns hydrologically adjusted elevation map which meet the criterium that
        all cells have an equal or lower elevation than its upstream neighboring cell.
        The function follows the algirithm by Yamazaki et al. (2012) to perform this 
        conditioning with minimal changes to the original elevation map.
        
        Arguments:
            elevtn {np.ndarray(float)} -- Elevation map
        
        Keyword Arguments:
            copy {bool} -- If True: copy the input map (default: {True})
        
        Raises:
            ValueError: The shape of the elevation and flow direction map do not match.
        
        Returns:
            np.ndarray -- Hydrologically adjusted elevation map 
        """
        if not np.all(self.shape == elevtn.shape):
            raise ValueError(f'The shape of the elevation and flow direction map do not match.')
        if copy:
            elevtn_new = np.copy(elevtn).ravel()
        else:
            elevtn_new = elevtn.ravel()
        idxs_ds = self.get_pits()
        return dem.hydrologically_adjust_elevation(idxs_ds, self._data_flat, elevtn_new, self.shape)
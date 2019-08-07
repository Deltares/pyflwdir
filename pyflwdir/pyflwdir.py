# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

import logging
import numpy as np
from numba import njit, jitclass
import rasterio
from rasterio.transform import Affine, array_bounds

# local
from .core import fd
from . import flux, utils, features, network, gridtools
# export
__all__ = ['FlwdirRaster']

# 
logger = logging.getLogger(__name__)

# global variabels
IDENTITY_NS = Affine(1, 0, 0, 0, -1, 0)
_ds = fd._ds
_us = fd._us 
_nodata = fd._nodata 
_pits = fd._pits

class FlwdirRaster(object):

    def __init__(self, data, 
            transform=IDENTITY_NS, crs=4326,
            check_format=False, create_copy=False):    
        assert data.ndim == 2
        assert np.sign(transform[4]) == -1
        # flwdir format props
        self.d8format = 'd8'
        self._ds = _ds 
        self._us = _us
        self._nodata = _nodata
        self._pits = _pits
        # data
        self.crs = rasterio.crs.CRS.from_user_input(crs)
        self.latlon = self.crs.is_geographic # NOTE: we assume this is in latlon
        self.transform = transform
        self.shape = data.shape
        self.bounds = array_bounds(data.shape[0], data.shape[1], transform)
        self.size = data.size
        if create_copy:
            self.data = data.copy()
        else:
            self.data = data.view() # view of data
        self.data_flat = self.data.ravel() # flattened view of data
        if check_format and fd._check_format(self.data) == False: # simple check. isvalid includes more rigorous check
            raise ValueError('Unknown flow direction values found in data')

    def __getitem__(self, key):
        return self.data_flat[key]

    def __setitem__(self, key, item):
        self.data_flat[key] = item

    def _xycoords(self):
        res = self.transform[0]
        xmin, ymin, xmax, ymax = self.bounds
        ys = np.linspace(ymax-res/2., ymin+res/2., self.shape[0])
        xs = np.linspace(xmin+res/2., xmax-res/2., self.shape[1])
        return xs, ys

    def isvalid(self):
        # check if all cells connect to pit / outflows at bounds
        valid = utils.flwdir_check(self.data)[1] == False
        return valid

    def repair(self):
        repair_idx, _ = utils.flwdir_check(self.data)
        if repair_idx.size > 0:
            self.data_flat[repair_idx] = self._pits[-1] # set inland pit

    def setup_network(self, idx0=None):
        if idx0 is None:
            idx0 = self.get_pits()
            if idx0.size == 0:
                raise ValueError('no pits found in flow direction data')
        self.idx0 = np.atleast_1d(idx0) # basin outlets
        self.rnodes, self.rnodes_up, self.rbasins = network.setup_dd(self.data_flat, self.idx0, self.shape)

    def get_pits(self):
        return fd.pit_indices(self.data.flatten())

    def upstream_area(self, cell_area=None):
        """returns upstream area in km"""
        if not hasattr(self, 'rnodes'):
            self.setup_network() # setup network based on pits
        if cell_area is None and self.latlon:
            cell_area = gridtools.latlon_cellare_metres(self.transform, self.shape)/1e6
        elif cell_area is None:
            cell_area = gridtools.cellare_metres(self.transform, self.shape)/1e6
        if not (np.all(cell_area.shape == self.shape)):
            raise ValueError(f"cell_area shape {cell_area.shape} does not match flwdir shape {self.shape}")
        return flux.propagate_downstream(self.rnodes, self.rnodes_up, material=cell_area)

    def basin_bounds(self):
        res = self.transform[0]
        xs, ys = self._xycoords()
        if not hasattr(self, 'rnodes'):
            self.setup_network() # setup network based on pits
        return features.basin_bbox(self.rnodes, self.rnodes_up, self.rbasins, ys, xs, res)

    def basin_map(self, idx=None, values=None, dtype=np.int64):
        if not hasattr(self, 'rnodes'):
            self.setup_network()    # setup network based on pits
        if idx is None:             # use outlet points if idx not given
            idx = self.idx0
        idx = np.atleast_1d(idx).astype(np.int64)
        if values is None:          # use range to number basins if not given
            values = np.arange(idx.size, dtype=dtype)+1
        else:
            values = np.atleast_1d(values).astype(dtype)
            if np.any(values<=0):
                raise ValueError('all values should be larger than zero')
        if not idx.size == values.size and idx.ndim == 1:
            raise ValueError('idx and values should be 1d arrays of same size')
        basidx_flat = np.zeros(self.size, dtype=dtype)
        basidx_flat[idx] = values
        basidx_flat = network.delineate_basins(self.rnodes, self.rnodes_up, basidx_flat=basidx_flat)
        return basidx_flat.reshape(self.shape)

    def basin_shape(self, basin_map=None, **kwargs):
        if basin_map is None:
            basin_map = self.basin_map(**kwargs)
        return gridtools.vectorize(basin_map, 0, self.transform, crs=self.crs)

    def stream_order(self):
        if not hasattr(self, 'rnodes'):
            self.setup_network() # setup network based on pits
        return network.stream_order(self.rnodes, self.rnodes_up, self.shape).astype(np.int16)

    def stream_shape(self, stream_order=None, mask=None, min_order=3):
        if stream_order is None:
            stream_order = self.stream_order()
        if mask is None:
            stream_order[stream_order < min_order] = np.int16(-1)
        else:
            stream_order[mask] = np.int16(-1)
        riv_nodes, riv_order = features.river_nodes(self.idx0, self.data_flat, stream_order.ravel(), self.shape)
        xs, ys = self._xycoords()
        geoms = [gridtools.nodes_to_ls(nodes, ys, xs, self.shape) for nodes in riv_nodes]
        return gridtools.gp.GeoDataFrame(data=riv_order, columns=['stream_order'], geometry=geoms, crs=self.crs)
        
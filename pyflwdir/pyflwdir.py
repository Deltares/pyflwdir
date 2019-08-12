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
from . import flux, utils, features, network, gridtools, d8_scaling
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
        self.res = transform[0], transform[4]
        self.cellare = np.abs(transform[0]*transform[4])
        self.shape = data.shape
        self.bounds = array_bounds(data.shape[0], data.shape[1], transform)
        self.size = data.size
        if self.size > 2**32-2:
            raise ValueError('Extent too large for uint32 indices')
        if create_copy:
            self._data = data.copy()
        else:
            self._data = data.view() # view of data
        self._data_flat = self._data.reshape(-1) # flattened view of data
        if check_format and fd._check_format(self._data) == False: # simple check. isvalid includes more rigorous check
            raise ValueError('Unknown flow direction values found in data')
        # set placeholder properties
        self._idx0 = None            # most downstream indices in network
        self._rnodes = None          # network ds nodes (n)
        self._rnodes_up = None       # network us nodes (n,m); m <= 8 in d8

    def __getitem__(self, key):
        return self._data_flat[key]

    def __setitem__(self, key, item):
        self._data_flat[key] = item

    # def __del__(self):
    #     del self._rnodes[:]
    #     del self._rnodes
    #     del self._rnodes_up[:]
    #     del self._rnodes_up

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
        # check if all cells connect to pit / outflows at bounds
        valid = utils.flwdir_check(self._data)[1] == False
        return valid

    def repair(self):
        repair_idx, _ = utils.flwdir_check(self._data)
        if repair_idx.size > 0:
            self._data_flat[repair_idx] = self._pits[-1] # set inland pit
        return None

    def setup_network(self, idx0=None):
        if idx0 is None:
            idx0 = self.get_pits()
            if idx0.size == 0:
                raise ValueError('no pits found in flow direction data')       
        self._idx0 = np.atleast_1d(np.asarray(idx0, dtype=np.uint32)) # basin outlets
        self._rnodes, self._rnodes_up = network.setup_dd(self._idx0, self._data_flat, self.shape)
        return None

    def get_pits(self):
        return fd.pit_indices(self._data.flatten())

    def upstream_area(self, cell_area=None):
        """returns upstream area in km"""
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        # if cell_area is None and self.latlon:
        #     cell_area = gridtools.latlon_cellare_metres(self.transform, self.shape)/1e6
        # elif cell_area is None:
        #     cell_area = gridtools.cellare_metres(self.transform, self.shape)/1e6
        # if not (np.all(cell_area.shape == self.shape)):
        #     raise ValueError(f"cell_area shape {cell_area.shape} does not match flwdir shape {self.shape}")
        # return flux.propagate_downstream(self._rnodes, self._rnodes_up, material=cell_area)
        if self.latlon:
            _, ys = self._xycoords()
            cellare = gridtools.lat_to_area(ys)*self.cellare
        else:
            cellare = np.ones(self.shape[0])*self.cellare
        return network.upstream_area(self._rnodes, self._rnodes_up, cellare/1e6, self.shape)

    def basin_bounds(self, idx=None):
        resx, resy = self.res
        xs, ys = self._xycoords()
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        if idx is None:             # use most downstream network indices if idx not given
            idx = self._idx0
        idx = np.atleast_1d(idx).astype(np.uint32)
        return features.basin_bbox(self._rnodes, self._rnodes_up, idx, ys, xs, resy, resx)

    def basin_map(self, idx=None, values=None, dtype=np.int32):
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        if idx is None:             # use most downstream network indices if idx not given
            idx = self._idx0
        idx = np.atleast_1d(idx).astype(np.uint32)
        if values is None:          # number basins using range, starting from 1
            values = np.arange(idx.size, dtype=dtype)+1
        else:
            values = np.atleast_1d(values).astype(dtype)
            if np.any(values<=0):
                raise ValueError('all values should be larger than zero')
        if not idx.size == values.size and idx.ndim == 1:
            raise ValueError('idx and values should be 1d arrays of same size')
        return network.delineate_basins(self._rnodes, self._rnodes_up, idx, values, self.shape)

    def subbasin_map_grid(self, scale_ratio, uparea=None):
        if not self.shape[0] % scale_ratio == self.shape[1] % scale_ratio == 0:
            raise ValueError(f'the data shape should be an exact multiplicity of the scale_ratio')
        if uparea == None:
            uparea = self.upstream_area()
        subbasin_idx = d8_scaling.subbasin_outlets_grid(scale_ratio, self._data, uparea)
        subbasin_idx = subbasin_idx[subbasin_idx!=-9999]
        return self.basin_map(idx=subbasin_idx, values=None, dtype=np.uint32)

    def subbasin_map_pfaf(self):
        # TODO create pfafstetter subbasins
        raise NotImplementedError()

    def basin_shape(self, basin_map=None, nodata=0, **kwargs):
        if basin_map is None:
            basin_map = self.basin_map(**kwargs)
            nodata = 0 # overwrite nodata
        return gridtools.vectorize(basin_map, nodata, self.transform, crs=self.crs)

    def stream_order(self):
        if self._rnodes is None:
            self.setup_network()    # setup network, with pits as most downstream indices
        return network.stream_order(self._rnodes, self._rnodes_up, self.shape)

    def stream_shape(self, stream_order=None, mask=None, min_order=3):
        if stream_order is None:
            stream_order = self.stream_order()
        if mask is None:
            stream_order[stream_order < min_order] = np.int16(-1)
        else:
            stream_order[mask] = np.int16(-1)
        riv_nodes, riv_order = features.river_nodes(self._idx0, self._data_flat, stream_order.reshape(-1), self.shape)
        xs, ys = self._xycoords()
        geoms = [gridtools.nodes_to_ls(nodes, ys, xs, self.shape) for nodes in riv_nodes]
        return gridtools.gp.GeoDataFrame(data=riv_order, columns=['stream_order'], geometry=geoms, crs=self.crs)
        
    def upscale(self, scale_ratio, uparea=None, upa_min=0.5, method='extended', return_outlets=False):
        if not self.shape[0] % scale_ratio == self.shape[1] % scale_ratio == 0:
            raise ValueError(f'the data shape should be an exact multiplicity of the scale_ratio')
        if uparea == None:
            uparea = self.upstream_area()
        transform_lr = self.transform.copy()
        transform_lr[0], transform_lr[4] = self.transform[0] * scale_ratio, self.transform[4] * scale_ratio
        flwdir_lr, outlet_lr = d8_scaling(scale_ratio, self._data, uparea, upa_min=upa_min, extended=method=='extended')
        flwdir_lr = FlwdirRaster(flwdir_lr, transform=transform_lr, crs=self.crs)
        return flwdir_lr, outlet_lr




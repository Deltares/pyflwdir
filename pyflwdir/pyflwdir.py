# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

""""""
import numpy as np
from numba import njit
from copy import deepcopy

# local
from pyflwdir import (
    core, core_d8, core_flow, core_ldd, network, basins, gis_utils
)
# export
__all__ = ['FlwdirRaster']

# import logging
# logger = logging.getLogger(__name__)
def flwdir_to_idxs(flwdir, ftype='d8', check_ftype=True):
    if ftype == 'd8':
        if check_ftype and not core_d8._is_d8(flwdir):
            raise ValueError('The flow direction type is not recognized as "d8".')
        idxs = core_d8.d8_to_idxs(flwdir)
    elif ftype == 'flow':
        if check_ftype and not core_flow._is_flow(flwdir):
            raise ValueError('The flow direction type is not recognized as "flow".')
        idxs = core_flow.flow_to_idxs(*flwdir)
    return idxs

def infer_ftype(flwdir):
    """infer flowdir type from data"""
    if core_d8._is_d8(flwdir):
        ftype = 'd8'
    elif core_flow._is_flow(flwdir):
        ftype = 'flow'
    elif core_ldd._is_ldd(flwdir):
        ftype = 'ldd'
    else:
        raise ValueError('The flow direction type not recognized.')
    return ftype

class FlwdirRaster(object):

    def __init__(
        self, 
        data, 
        ftype='d8',
        check_ftype=True,
    ):
        """Create an instance of a pyflwdir.FlwDirRaster object"""
        if ftype == 'infer':
            ftype = infer_ftype(data)
        if ftype == 'flow':
            if not isinstance(data, tuple) and not (data.ndim == 3 and data.shape[0] == 2):
                raise ValueError(
                    'Data should be a tuple(nextx,nexty) or array with shape(2,height,width) for the "flow" flow direction type.'
                    )
            size = data[0].size
            shape = data[0].shape
        else:
            size = data.size 
            shape = data.shape
        
        # set network
        if size > 2**32-2:       # maximum size we can have with uint32 indices
            raise ValueError("The current framework limits the raster size to 2**32-2 cells")
        self.ftype = ftype
        self.shape = shape
        # convert to ds index array
        self._idxs = flwdir_to_idxs(data, ftype=ftype, check_ftype=check_ftype)
        
        # set placeholder network for network
        self._pits = None            # most downstream indices in network
        self._network = None         # Tuple(network ds nodes (n), network us nodes (n,m)) m <= 8 in d8

    @property
    def pits(self):
        if self._pits is None:
            self.set_pits()
        return self._pits

    @property
    def network(self):
        if self._network is None:
            self.set_network()    # setup network, with pits as most downstream indices
        return self._network

    def set_pits(self, idx=None, streams=None):
        if idx is None:
            _pits = core.pit_indices(self._idxs)
            if _pits.size == 0:
                raise ValueError('No pits found in data')
        else:
            _pits = np.asarray(idx, dtype=np.uint32).flatten()
            if np.any(_pits<0) or np.any(_pits>=self._idxs.size):
                raise  ValueError("Pit indices outside domain")
            if streams is not None: # snap to streams
                _pits = core.ds_stream(_pits, streams)
        self._pits = _pits
        self._network = None       # reset network tree

    def set_network(self, idx=None):
        """Setup network with upstream - downstream connections"""
        if idx is not None:
            self.set_pits(idx=idx)
        self._network = network.setup_network(self._idxs, self.pits)

    def isvalid(self):
        """Returns True if the flow direction map is valid"""
        return core.error_indices(self._idxs).size == 0

    def repair(self):
        """Repair the flow direction map in order to have all cells flow to a pit."""
        repair_idx = core.error_indices(self._idxs)
        if repair_idx.size > 0:
            self._idxs[repair_idx] = repair_idx

    def upstream_area(self, latlon=False, affine=gis_utils.IDENTITY):
        """Returns the upstream area based on the flow direction map. """
        return network.upstream_area(*self.network, self.shape, latlon=latlon, affine=affine)

    def stream_order(self):
        """Returns the Strahler Order map (TODO ref). 
        The smallest streams, which are the cells with no upstream cells, get an order 1. 
        Where two channels of order 1 join, a channel of order 2 results downstream. 
        In general, where two channels of order i join, a channel of order i+1 results.
        """
        return network.stream_order(*self.network, self.shape)

    def accuflux(self, material, nodata=-9999):
        """Return the accumulated"""
        if not np.all(material.shape == self.shape):
            raise ValueError("'Material' shape does not match with flow direction shape")
        return network.accuflux(*self.network, material, nodata)

    def basins(self):
        """Returns a 2d array with a unique id for every basin"""
        return basins.basins(*self.network, self.shape)
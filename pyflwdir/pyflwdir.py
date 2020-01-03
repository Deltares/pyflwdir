# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019

""""""
import numpy as np
# local
import pyflwdir
from pyflwdir import (
    core, 
    core_d8, 
    core_flow, 
    core_ldd, 
    network, 
    gis_utils, 
    basin_utils, 
    basin_descriptors
)
ftypes = {'d8':core_d8, 'ldd':core_ldd, 'flow':core_flow}

# export
__all__ = ['FlwdirRaster']

# import logging
# logger = logging.getLogger(__name__)
def flwdir_to_idxs(flwdir, ftype='d8', check_ftype=True):
    fd = ftypes[ftype]
    if check_ftype and not fd.isvalid(flwdir):
        raise ValueError(f'The flow direction type is not recognized as "{ftype}".')
    return fd.from_flwdir(flwdir)

def infer_ftype(flwdir):
    """infer flowdir type from data"""
    ftype = None
    for fd in ftypes:
        if fd.isvalid(flwdir):
            ftype = fd._ftype
            break
    if ftype is None:
        raise ValueError('The flow direction type not recognized.')
    return ftype

class FlwdirRaster(object):

    def __init__(
        self, 
        data, 
        ftype='d8',
        check_ftype=True,
        # _max_depth=None,
    ):
        """Create an instance of a pyflwdir.FlwDirRaster object"""
        if ftype == 'infer':
            ftype = infer_ftype(data)
        if ftype == 'flow':
            # if not isinstance(data, tuple) and not (data.ndim == 3 and data.shape[0] == 2):
            #     raise ValueError(
            #         'Data should be a tuple(nextx,nexty) or array with shape(2,height,width) for the "flow" flow direction type.'
            #         )
            self.size = data[0].size
            self.shape = data[0].shape
            # _max_depth = 35 if _max_depth is None else _max_depth
        else:
            self.size = data.size 
            self.shape = data.shape
        if self.size > 2**32-2:       # maximum size we can have with uint32 indices
            raise ValueError("The current framework limits the raster size to 2**32-2 cells")
        if len(self.shape) != 2:
            raise ValueError("Flow direction array should be 2 dimensional")
        self.ftype = ftype
        self._core = ftypes[ftype]

        # initialize
        # convert to internal indices
        self._idxs_valid, self._idxs_ds, self._idxs_us, self._idx0 = flwdir_to_idxs(
            data, 
            ftype=ftype, 
            check_ftype=check_ftype, 
            # _max_depth=_max_depth
        )
        if np.any([s <= 1 for s in self.shape]) or self._idxs_valid.size <= 1:
            raise ValueError('Invalid flow direction raster: too small (length or width equal to one)')
        elif self._idx0.size == 0:
            raise ValueError('Invalid flow direction raster: no pits found')

        # set placeholder for network tree
        self._tree = None         # List of array ordered from down- to upstream

    @property
    def pits(self):
        """return flattened (1D) raster index of pits"""
        return self._idxs_valid[self._idx0]

    @property
    def _pits(self):
        """return internal index of pits"""
        return self._idx0

    @property
    def _network(self):
        """return tree (list of 1D arrays) and upstream indices (2D array) with internal indices"""
        if self._tree is None:
            self.set_network()    # setup network, with pits as most downstream indices
        return self._tree, self._idxs_us

    def _reshape(self, data, nodata):
        """return 2D array from 1D data at valid indices and filled with nodata"""
        return core._reshape(data, self._idxs_valid, self.shape, nodata=nodata)

    def _flatten(self, data):
        """return 1D data array at valid indices for internal operations"""
        return data.flat[self._idxs_valid]

    def _internal_idx(self, idx):
        """return interal index based on flattened (1D) raster index"""
        return core.internal_idx(idx, self._idxs_valid, self.size)

    def set_pits(self, idx, streams=None):
        """Set pits for usage in downstream funtions"""
        idx = np.asarray(idx, dtype=np.uint32).flatten()
        _pits = self._internal_idx(idx)
        if streams is not None: # snap to streams
            _pits = core.ds_stream(_pits, self._idxs_ds, self._flatten(streams))
        self._idx0 = _pits
        self._tree = None       # reset network tree

    def set_network(self, idx=None):
        """Setup network with upstream - downstream connections"""
        if idx is not None:
            self.set_pits(idx=idx)
        self._tree = network.setup_network(self._pits, self._idxs_ds, self._idxs_us)

    def isvalid(self):
        """Returns True if the flow direction map is valid"""
        return core.error_indices(self._pits, self._idxs_ds, self._idxs_us).size == 0

    def repair(self):
        """Repair the flow direction map in order to have all cells flow to a pit."""
        repair_idx = core.error_indices(self._pits, self._idxs_ds, self._idxs_us)
        if repair_idx.size > 0:
            self._idxs_ds[repair_idx] = repair_idx

    def upstream_area(self, latlon=False, affine=gis_utils.IDENTITY):
        """Returns the upstream area based on the flow direction map. """
        upa_flat = network.upstream_area(*self._network, self.shape, latlon=latlon, affine=affine)
        return self._reshape(upa_flat, nodata=-9999.)

    def stream_order(self):
        """Returns the Strahler Order map (TODO ref). 
        The smallest streams, which are the cells with no upstream cells, get an order 1. 
        Where two channels of order 1 join, a channel of order 2 results downstream. 
        In general, where two channels of order i join, a channel of order i+1 results.
        """
        return self._reshape(network.stream_order(*self._network), nodata=np.int8(-1))

    def accuflux(self, material, nodata=-9999):
        """Return the material array accumulated along the flow direction map."""
        if not np.all(material.shape == self.shape):
            raise ValueError("'Material' shape does not match with flow direction shape")
        accu_flat = network.accuflux(*self._network, self._flatten(material), nodata)
        return self._reshape(accu_flat, nodata=nodata)

    def basins(self):
        """Returns a 2d array with a unique id for every basin"""
        return self._reshape(network.basins(*self._network), nodata=np.uint32(0))

    def drainage_path_stats(self, rivlen, elevtn):
        if not np.all(rivlen.shape == self.shape):
            raise ValueError("'rivlen' shape does not match with flow direction shape")
        if not np.all(elevtn.shape == self.shape):
            raise ValueError("'elevtn' shape does not match with flow direction shape")
        df_out = basin_descriptors.mean_drainage_path_stats(*self._network, 
            self._flatten(rivlen), self._flatten(elevtn))
        return df_out
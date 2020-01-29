# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
""""""

import numpy as np
from pyflwdir import (
    basin,
    basin_utils,
    basin_descriptors,
    core,
    core_d8,
    core_nextxy,
    core_nextidx,
    core_ldd,
    gis_utils,
)
ftypes = {
    core_d8._ftype: core_d8,
    core_ldd._ftype: core_ldd,
    core_nextxy._ftype: core_nextxy,
    core_nextidx._ftype: core_nextidx
}

# export
__all__ = ['FlwdirRaster']


# import logging
# logger = logging.getLogger(__name__)
def parse_flwdir(flwdir, ftype, check_ftype=True):
    fd = ftypes[ftype]
    if check_ftype and not fd.isvalid(flwdir):
        raise ValueError(
            f'The flow direction type is not recognized as "{ftype}".')
    return fd.from_flwdir(flwdir)


def infer_ftype(flwdir):
    """infer flowdir type from data"""
    ftype = None
    for _, fd in ftypes.items():
        if fd.isvalid(flwdir):
            ftype = fd._ftype
            break
    if ftype is None:
        raise ValueError('The flow direction type not recognized.')
    return ftype


class FlwdirRaster(object):
    """
    Flow direction raster array parsed to actionable format.

    Attributes
    ----------
    size : int
        flwdir raster size
    shape : tuple of int
        flwdir raster shape
    ncells : int
        number of valid cells
    ftype : {'d8', 'ldd', 'nextxy'}
        flow direction type
    pits : ndarray of int
        1D raster indices of pit cells
    tree : list of ndarray of int 
        indices ordered from down- to upstream
    isvalid : bool
        True if no loops in flow direction data
    
    Methods
    ------
    set_pits
    set_tree
    basins
    repair
    upstream_area
    stream_order
    accuflux
    
    """
    def __init__(
        self,
        data,
        ftype='infer',
        check_ftype=True,
        # _max_depth=None,
    ):
        """Flow direction raster array parsed to actionable format.
        
        Parameters
        ----------
        data : ndarray
            flow direction raster data
        ftype : {'d8', 'ldd', 'nextxy', 'infer'}, optional
            name of flow direction type, infer from data if 'infer'
            (the default is 'infer')
        check_ftype : bool, optional
            check if valid flow direction raster, only if ftype is not 'infer'
            (the default is True)
        """
        if ftype == 'infer':
            ftype = infer_ftype(data)
            check_ftype = False  # already done
        if ftype == 'nextxy':
            self.size = data[0].size
            self.shape = data[0].shape
            # _max_depth = 35 if _max_depth is None else _max_depth
        else:
            self.size = data.size
            self.shape = data.shape

        if self.size > 2**32 - 2:  # maximum size we can have with uint32 indices
            raise ValueError(
                "The current framework limits the raster size to 2**32-2 cells"
            )
        if len(self.shape) != 2:
            raise ValueError("Flow direction array should be 2 dimensional")

        self.ftype = ftype
        self._core = ftypes[ftype]

        # initialize
        # convert to internal indices
        self._idxs_valid, self._idxs_ds, self._idxs_us, self._idx0 = parse_flwdir(
            data,
            ftype=ftype,
            check_ftype=check_ftype,
            # _max_depth=_max_depth
        )
        if self._idxs_valid.size <= 1:
            raise ValueError(
                'Invalid flow direction raster: size equal or smaller to 1')
        elif self._idx0.size == 0:
            raise ValueError('Invalid flow direction raster: no pits found')
        self.ncells = self._idxs_valid.size
        # set placeholder for network tree
        self._tree = None  # List of array ordered from down- to upstream

    @property
    def pits(self):
        """Return 1D raster indices of pits"""
        return self._idxs_valid[self._idx0]

    @property
    def tree(self):
        """Return network tree, a list of arrays ordered from down- to upstream"""
        if self._tree is None:
            self.set_tree(
            )  # setup network, with pits as most downstream indices
        return self._tree

    @property
    def isvalid(self):
        """Returns True if the flow direction map is valid."""
        return core.loop_indices(self._idxs_ds, self._idxs_us).size == 0

    def set_pits(self, idxs_pit, streams=None):
        """Reset original pits from the flow direction raster based on `idxs_pit`.
        If streams is given, the pits are moved to the first downstream 'stream' cell.
        
        Parameters
        ----------
        idxs_pit : array_like, optional
            raster 1D indices of pits
            (the default is None, in which case the pits are infered from flwdir data)
        streams : ndarray of bool, optional
            2D raster with cells flagged 'True' at river/stream cells  
            (the default is None) 
        """
        idxs_pit = np.asarray(idxs_pit, dtype=np.uint32).flatten()
        idxs_pit = self._internal_idx(idxs_pit)
        if streams is not None:  # snap to streams
            idxs_pit = core.ds_stream(idxs_pit, self._idxs_ds,
                                      self._flatten(streams))
        self._idx0 = idxs_pit
        self._tree = None  # reset network tree

    def set_tree(self, idxs_pit=None, streams=None):
        """Setup network tree, a list of arrays ordered from down- to upstream.
        If idxs_pit is given, the tree is setup cells upstream from these points only,
        ignoring the pits in the original flow direction raster.
        See `set_pits` for info about the parameters of the method. 
        """
        if idxs_pit is not None:
            self.set_pits(idxs_pit=idxs_pit, streams=streams)
        self._tree = core.network_tree(self._idx0, self._idxs_us)

    def repair(self):
        """Set pits at cells witch do not drain to a pit."""
        repair_idx = core.loop_indices(self._idxs_ds, self._idxs_us)
        if repair_idx.size > 0:
            # set pits for all loop indices !
            self._idxs_ds[repair_idx] = repair_idx
            self._idx0 = core.pit_indices[self._idxs_ds]

    def basins(self):
        """Returns a basin map with a unique id (starting from 1) for every basin"""
        return self._reshape(basin.basins(self.tree, self._idxs_us,
                                          self.tree[0]),
                             nodata=np.uint32(0))

    def subbasins(self, idxs):
        """Returns a subbasin map with a unique id (starting from 1) for every subbasin"""
        return self._reshape(basin.basins(self.tree, self._idxs_us,
                                          self._internal_idx(idxs)),
                             nodata=np.uint32(0))

    def upstream_area(self,
                      affine=gis_utils.IDENTITY,
                      latlon=False,
                      nodata=-9999.):
        """Returns the upstream area map based on the flow direction map. 


        Parameters
        ----------
        latlon : bool, optional
            True if WGS84 coordinates
            (the default is False)
        affine : affine transform
            Two dimensional affine transform for 2D linear mapping
            (the default is an identity transform which results in an area of 1 for every cell)
        nodata : int or float
            Missing data value for cells outside domain

        Returns
        -------
        upstream area map: 2D array
        """
        upa_flat = basin.upstream_area(self.tree,
                                       self._idxs_valid,
                                       self._idxs_us,
                                       self.shape[1],
                                       affine=affine,
                                       latlon=latlon)
        return self._reshape(upa_flat, nodata=nodata)

    def stream_order(self):
        """Returns the Strahler Order map [1]. 

        The smallest streams, which are the cells with no upstream cells, get an order 1. 
        Where two channels of order 1 join, a channel of order 2 results downstream. 
        In general, where two channels of order i join, a channel of order i+1 results.

        .. [1] Strahler, A.N., 1964, Quantitative geomorphology of drainage basins and channel networks, 
        section 4-II. In: Handbook of Applied Hydrology (V.T. Chow, et al. (1988)), McGraw-Hill, New York USA

        Returns
        -------
        strahler order map: 2D array of int
        """
        return self._reshape(basin.stream_order(self.tree, self._idxs_us),
                             nodata=np.int8(-1))

    def accuflux(self, material, nodata=-9999):
        """Return accumulated amount of material along the flow direction map.
        

        Parameters
        ----------
        material : 2D array
            2D raster with initial amount of material
        nodata : int or float
            Missing data value for cells outside domain

        Returns
        -------
        accumulated material map: 2D array
        """
        if not np.all(material.shape == self.shape):
            raise ValueError(
                "'Material' shape does not match with flow direction shape.")
        accu_flat = basin.accuflux(self.tree, self._idxs_us,
                                   self._flatten(material), nodata)
        return self._reshape(accu_flat, nodata=nodata)

    def vector(self,
               min_order=1,
               xs=None,
               ys=None,
               affine=gis_utils.IDENTITY,
               crs=None):
        """Returns a GeoDataFrame with river segments. Segments are LineStrings 
        connecting cell from down to upstream and are splitted based on strahler order. 
        
        The coordinates of the nodes are based on the cell center as calculated using the
        affine transform, unless maps with x and y coordinates are provided.
        
        Parameters
        ----------
        min_order : int
            Minimum Strahler Order recognized as river 
            (the default is 1)
        xs, ys : ndarray of float
            Raster with cell x, y coordinates 
            (the default is None, taking the cell center coordinates)
        affine : affine transform
            Two dimensional affine transform for 2D linear mapping
            (the default is an identity transform)
        crs : coordinate reference system

        
        Returns
        -------
        river segments : geopandas.GeoDataFrame
        """
        try:
            import geopandas as gp
            import shapely
        except ImportError:
            raise ImportError(
                "The `vector` method requies the shapely and geopandas packages"
            )
        # get geoms and make geopandas dataframe
        if xs is None or ys is None:
            xs, ys = gis_utils.idxs_to_coords(self._idxs_valid, affine,
                                              self.shape)
        else:
            xs, ys = xs.ravel()[self._idxs_valid], ys.ravel()[self._idxs_valid]
        geoms = gis_utils.idxs_to_geoms(self._idxs_ds, xs, ys)
        gdf = gp.GeoDataFrame(geometry=geoms, crs=crs)
        # get additional data
        gdf['stream_order'] = self.stream_order().flat[self._idxs_valid]
        gdf['pit'] = self._idxs_ds == np.arange(self.ncells)
        return gdf

    def drainage_path_stats(self, rivlen, elevtn):
        if not np.all(rivlen.shape == self.shape):
            raise ValueError(
                "'rivlen' shape does not match with flow direction shape")
        if not np.all(elevtn.shape == self.shape):
            raise ValueError(
                "'elevtn' shape does not match with flow direction shape")
        df_out = basin_descriptors.mean_drainage_path_stats(
            self.tree, self._idxs_us, self._flatten(rivlen),
            self._flatten(elevtn))
        return df_out

    def _reshape(self, data, nodata):
        """Return 2D array from 1D data at valid indices and filled with nodata."""
        return core._reshape(data, self._idxs_valid, self.shape, nodata=nodata)

    def _flatten(self, data):
        """Return 1D data array at valid indices for internal operations."""
        return data.flat[self._idxs_valid]

    def _internal_idx(self, idx):
        """Return interal indices based on 1D raster indices."""
        # NOTE: should we throw an error if any idx is invalid ?
        return core._internal_idx(idx, self._idxs_valid, self.size)

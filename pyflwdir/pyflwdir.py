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
    subgrid,
    upscale,
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
    return fd.from_array(flwdir)


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
    ftype : {'d8', 'ldd', 'nextxy', 'nextidx'}
        flow direction type
    pits : ndarray of int
        1D raster indices of pit cells
    tree : list of ndarray of int 
        indices ordered from down- to upstream
    isvalid : bool
        True if no loops in flow direction data
    
    Methods
    ------
    `set_pits`
    `set_tree`
    `basins`
    `repair`
    `upstream_area`
    `stream_order`
    `accuflux`
    
    """
    def __init__(
        self,
        data,
        ftype='infer',
        check_ftype=True,
    ):
        """Flow direction raster array parsed to actionable format.
        
        Parameters
        ----------
        data : ndarray
            flow direction raster data
        ftype : {'d8', 'ldd', 'nextxy', 'nextidx', 'infer'}, optional
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
        """Returns 1D raster indices of pits
        
        Returns
        -------
        1D array of int
            1D raster indices of pit
        """
        return self._idxs_valid[self._idx0]

    @property
    def tree(self):
        """Returns the network tree: a list of arrays ordered from down- to 
        upstream. 
        
        Returns
        -------
        list of ndarray of int
            river network tree
        """
        if self._tree is None:
            self.set_tree(
            )  # setup network, with pits as most downstream indices
        return self._tree

    @property
    def isvalid(self):
        """Returns True if the flow direction map is valid."""
        return core.loop_indices(self._idxs_ds, self._idxs_us).size == 0

    def set_pits(self, idxs_pit, streams=None):
        """Reset original pits from the flow direction raster based on 
        `idxs_pit`.
        
        If streams is given, the pits are moved to the first downstream 
        `stream` cell.
        
        Parameters
        ----------
        idxs_pit : array_like, optional
            raster 1D indices of pits
            (the default is None, in which case the pits are infered 
            from flwdir data)
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
        """Setup the network tree: a list of arrays ordered from down- to 
        upstream. 
        
        If idxs_pit is given, the tree is setup cells upstream from these 
        points only, ignoring the pits in the original flow direction raster.
        If streams is given, the pits are moved to the first downstream 
        `stream` cell.

        Parameters
        ----------
        idxs_pit : array_like, optional
            raster 1D indices of pits
            (the default is None, in which case the pits are infered 
            from flwdir data)
        streams : ndarray of bool, optional
            2D raster with cells flagged 'True' at river/stream cells  
            (the default is None) 
        """
        if idxs_pit is not None:
            self.set_pits(idxs_pit=idxs_pit, streams=streams)
        self._tree = core.network_tree(self._idx0, self._idxs_us)

    def repair(self):
        """Repairs loops by set a pit at every cell witch does not drain to 
        a pit."""
        repair_idx = core.loop_indices(self._idxs_ds, self._idxs_us)
        if repair_idx.size > 0:
            # set pits for all loop indices !
            self._idxs_ds[repair_idx] = repair_idx
            self._idx0 = core.pit_indices[self._idxs_ds]

    def to_array(self, ftype=None):
        """Return 2D flow direction array. 
        
        Parameters
        ----------
        ftype : {'d8', 'ldd', 'nextxy', 'nextidx'}, optional
            name of flow direction type
            (the default is None; use input ftype)
        
        Returns
        -------
        2D array of int
            Flow direction raster
        """
        if ftype is None:
            ftype = self.ftype
        if ftype not in ftypes:
            msg = f'The flow direction type "{ftype}" is not recognized.'
            raise ValueError(msg)
        return ftypes[ftype].to_array(self._idxs_valid, self._idxs_ds, 
                                      self.shape)

    def basins(self):
        """Returns a basin map with a unique IDs for every basin. The IDs
        start from 1 and the background value is 0.
        
        Returns
        -------
        ndarray of uint32
            basin map
        """
        basids = basin.basins(self.tree, self._idxs_us, self.tree[0])
        return self._reshape(basids, nodata=np.uint32(0))

    def subbasins(self, idxs):
        """Returns a subbasin map with a unique ID for every subbasin. The IDs
        start from 1 and the background value is 0.
        
        Parameters
        ----------
        idxs : ndarray of int
            1D raster indices of subbasin outlets
        
        Returns
        -------
        2D array of uint32
            subbasin map
        """
        idxs0 = self._internal_idx(idxs)
        subbas = basin.basins(self.tree, self._idxs_us, idxs0)
        return self._reshape(subbas, nodata=np.uint32(0))

    def upstream_area(self, affine=gis_utils.IDENTITY, latlon=False):
        """Returns the upstream area map based on the flow direction map. 
        
        If latlon is True it converts the cell areas to metres, otherwise it
        assumes the coordinate unit is metres.
        
        Parameters
        ----------
        latlon : bool, optional
            True if WGS84 coordinates
            (the default is False)
        affine : affine transform
            Two dimensional affine transform for 2D linear mapping
            (the default is an identity transform; cell area = 1)

        Returns
        -------
        2D array of float
            upstream area map [m2]
        """
        uparea = basin.upstream_area(self.tree,
                                     self._idxs_valid,
                                     self._idxs_us,
                                     self.shape[1],
                                     affine=affine,
                                     latlon=latlon)
        return self._reshape(uparea, nodata=-9999.)

    def stream_order(self):
        """Returns the Strahler Order map [1]. 

        The smallest streams, which are the cells with no upstream cells, get 
        order 1. Where two channels of order 1 join, a channel of order 2 
        results downstream. In general, where two channels of order i join, 
        a channel of order i+1 results.

        .. [1] Strahler, A.N., 1964 "Quantitative geomorphology of drainage 
        basins and channel networks, section 4-II". In: Handbook of Applied 
        Hydrology (V.T. Chow, et al. (1988)), McGraw-Hill, New York USA

        Returns
        -------
        2D array of int
            strahler order map
        """
        strord = basin.stream_order(self.tree, self._idxs_us)
        return self._reshape(strord, nodata=np.int8(-1))

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
        2D array of `material` dtype
            accumulated material map
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
        connecting cell from down to upstream. 
        
        The coordinates of the nodes are based on the cell center as calculated 
        using the affine transform, unless maps with subgrid x and y 
        coordinates are provided.
        
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
        crs : str, optional 
            Coordinate reference system to use for the returned GeoDataFrame
        
        Returns
        -------
        geopandas.GeoDataFrame
            flow direction vector
        """
        try:
            import geopandas as gp
            import shapely
        except ImportError:
            raise ImportError("The `vector` method requires the additional " +
                              "shapely and geopandas packages.")
        # get geoms and make geopandas dataframe
        if xs is None or ys is None:
            xs, ys = gis_utils.idxs_to_coords(self._idxs_valid, affine,
                                              self.shape)
        else:
            if xs.size != self.size or ys.size != self.size:
                raise ValueError("'xs' and/or 'ys' size does not match with" +
                                 " flow direction size")
            xs, ys = xs.ravel()[self._idxs_valid], ys.ravel()[self._idxs_valid]
        geoms = gis_utils.idxs_to_geoms(self._idxs_ds, xs, ys)
        gdf = gp.GeoDataFrame(geometry=geoms, crs=crs)
        # get additional meta data
        gdf['stream_order'] = self.stream_order().flat[self._idxs_valid]
        gdf['pit'] = self._idxs_ds == np.arange(self.ncells)
        return gdf

    def upscale(self, scale_factor, method='cosm', uparea=None, **kwargs):
        """Upscale flow direction network to lower resolution. 
        Available methods are Connecting Outlets Scaling Method (COSM) [1], 
        Effective Area Method (EAM) [2] and Double Maximum Method (DMM) [3].

        Note: This method only works for D8 or LDD flow directon data.
        
        # TODO update ref
        ..[1] Eilander et al. in preperation
        ..[2] Yamazaki D, Masutomi Y, Oki T and Kanae S 2008 
        "An Improved Upscaling Method to Construct a Global River Map" APHW
        ..[3] Olivera F, Lear M S, Famiglietti J S and Asante K 2002 
        "Extracting low-resolution river networks from high-resolution digital 
        elevation models" Water Resour. Res. 38 13-1-13–8 
        Online: http://doi.wiley.com/10.1029/2001WR000726
    
        
        Parameters
        ----------
        scale_factor : int
            number gridcells in resulting upscaled gridcell
        method : {'cosm', 'eam', 'dmm'}
            upscaling method
            (by default 'cosm')
        uparea : 2D array of float, optional
            2D raster with upstream area 
            (by default None; calculated on the fly)

        Returns
        ------
        FlwdirRaster
            upscaled Flow Direction Raster
        ndarray of uint32
            1D raster indices of subgrid outlets
        """
        methods = ['cosm', 'eam', 'dmm']
        if self.ftype not in  ['d8', 'ldd']:
            raise ValueError("The upscale method only works for D8 or LDD " +
                             "flow directon data.")
        if method not in methods:
            methodstr = "', '".join(methods)
            raise ValueError(f"Unknown method, select from: '{methodstr}'")
        if uparea is None:
            uparea = self.upstream_area()
        elif not np.all(uparea.shape == self.shape):
            raise ValueError(
                "'uparea' shape does not match with flow direction shape")
        # upscale flow directions
        fupscale = getattr(upscale, method)
        nextidx, subidxs_out = fupscale(subidxs_ds = self._idxs_ds,
                                        subidxs_valid = self._idxs_valid,
                                        subuparea = uparea.ravel(),
                                        subshape = self.shape,
                                        cellsize = scale_factor,
                                        **kwargs)
        dir_lr = FlwdirRaster(nextidx, ftype='nextidx', check_ftype=False)
        if not dir_lr.isvalid:
            raise ValueError(
                'The upscaled flow direction network is invalid.' +
                'Please provide a minimal reproducible example.')
        return dir_lr, subidxs_out

    def subarea(self,
                other,
                subidxs_out,
                affine=gis_utils.IDENTITY,
                latlon=False):
        """Returns the subgrid cell area, which is the specific area draining 
        to the outlet of a cell.

        If latlon is True it converts the cell areas to metres, otherwise it
        assumes the coordinate unit is metres.
        
        Parameters
        ----------
        other : FlwdirRaster
            upscaled Flow Direction Raster
        subidxs_out : 2D array of int
            flattened raster indices of subgrid outlets
        affine : affine transform
            Two dimensional affine transform for 2D linear mapping
            (the default is an identity transform; cell area = 1)
        latlon : bool, optional
            True if WGS84 coordinates
            (the default is False)
        
        Returns
        -------
        2D array of float with other.shape
            subgrid cell area [m2]
        """
        subidxs_out0 = _check_convert_subidxs_out(subidxs_out, other)
        if np.any(subidxs_out0 == core._mv):
            raise ValueError("invalid 'subidxs_out' with missing values" +
                             "at valid indices.")
        subare = subgrid.cell_area(self._internal_idx(subidxs_out0),
                                   self._idxs_valid,
                                   self._idxs_us,
                                   self.shape,
                                   latlon=latlon,
                                   affine=affine)
        return other._reshape(subare, -9999.)

    def subriver(self,
                 other,
                 subidxs_out,
                 elevtn,
                 uparea=None,
                 min_uparea=0.,
                 latlon=False,
                 affine=gis_utils.IDENTITY):
        """Returns the subgrid river length and slope per lowres cell. The 
        subgrid river is defined by the path starting at the subgrid outlet 
        cell moving upstream following the upstream subgrid cells with the 
        largest upstream area until it reaches the next upstream outlet cell. 
        
        A mimumum upstream area can be set to discriminate river cells.

        If latlon is True it converts the cell areas to metres, otherwise it
        assumes the coordinate unit is metres.
        
        Parameters
        ----------
        other : FlwdirRaster
            upscaled Flow Direction Raster
        subidxs_out : 2D array of int
            flattened raster indices of subgrid outlets
        elevnt : 2D array of float
            elevation raster
        uparea : 2D array of float, optional
            2D raster with upstream area, if None it is calculated 
            (by default None)
        min_uparea : float, optional
            Minimum upstream area to be consided as stream. Only used if 
            return_sublength is True. 
            (by default 0.)
        latlon : bool, optional
            True if WGS84 coordinates
            (the default is False)
        affine : affine transform
            Two dimensional affine transform for 2D linear mapping
            (the default is an identity transform; cell length = 1)
        
        Returns
        -------
        2D array of float with other.shape
            subgrid river length [m]
        2D array of float with other.shape
            subgrid river slope [m/m]
        """
        subidxs_out0 = _check_convert_subidxs_out(subidxs_out, other)
        if not np.all(elevtn.shape == self.shape):
            raise ValueError(
                "'elevtn' shape does not match with flow direction shape")
        if uparea is None:
            uparea = self.upstream_area(latlon=latlon, affine=affine)
        elif not np.all(uparea.shape == self.shape):
            raise ValueError(
                "'uparea' shape does not match with flow direction shape")
        rivlen, rivslp = subgrid.river_params(
            subidxs_out=self._internal_idx(subidxs_out0),
            subidxs_valid=self._idxs_valid,
            subidxs_ds=self._idxs_ds,
            subidxs_us=self._idxs_us,
            subuparea=uparea.flat[self._idxs_valid],
            subelevtn=elevtn.flat[self._idxs_valid],
            subshape=self.shape,
            min_uparea=min_uparea,
            latlon=latlon,
            affine=affine)
        return other._reshape(rivlen, -9999.), other._reshape(rivslp, -9999.)

    def subconnect(self, other, subidxs_out):
        """Returns binary array with ones if sugrid outlet cells are connected 
        in d8.
        
        Parameters
        ----------
        other : FlwdirRaster
            upscaled Flow Direction Raster
        subidxs_out : 2D array of int
            flattened raster indices of subgrid outlets
        
        Returns
        -------
        2D array of bool with other.shape
            valid subgrid connection
        """
        subidxs_out0 = _check_convert_subidxs_out(subidxs_out, other)
        subcon = subgrid.connected(self._internal_idx(subidxs_out0),
                                   other._idxs_ds, self._idxs_ds)
        return other._reshape(subcon, True)

    # def drainage_path_stats(self, rivlen, elevtn):
    #     if not np.all(rivlen.shape == self.shape):
    #         raise ValueError(
    #             "'rivlen' shape does not match with flow direction shape")
    #     if not np.all(elevtn.shape == self.shape):
    #         raise ValueError(
    #             "'elevtn' shape does not match with flow direction shape")
    #     df_out = basin_descriptors.mean_drainage_path_stats(
    #         self.tree, self._idxs_us, self._flatten(rivlen),
    #         self._flatten(elevtn))
    #     return df_out

    def _reshape(self, data, nodata):
        """Return 2D array from 1D data at valid indices and filled with 
        nodata."""
        return core._reshape(data, self._idxs_valid, self.shape, nodata=nodata)

    def _flatten(self, data):
        """Return 1D data array at valid indices for internal operations."""
        return data.flat[self._idxs_valid]

    def _internal_idx(self, idx):
        """Return interal indices based on 1D raster indices."""
        # NOTE: should we throw an error if any idx is invalid ?
        return core._internal_idx(idx, self._idxs_valid, self.size)


def _check_convert_subidxs_out(subidxs_out, other):
    """Convert 2D subidxs_out grid to 1D array at valid indices"""
    if not isinstance(other, FlwdirRaster):
        raise ValueError(
            "'other' is not recognized as instance of FlwdirRaster")
    if not np.all(subidxs_out.shape == other.shape):
        raise ValueError("'subidxs_out' shape does not match with `other`" +
                         " flow direction shape")
    subidxs_out0 = subidxs_out.ravel()[other._idxs_valid]
    if np.any(subidxs_out0 == core._mv):
        raise ValueError("invalid 'subidxs_out' with missing values" +
                         "at valid indices.")
    return subidxs_out0

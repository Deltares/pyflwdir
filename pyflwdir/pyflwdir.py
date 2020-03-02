# -*- coding: utf-8 -*-
# Author: Dirk Eilander (contact: dirk.eilander@deltares.nl)
# August 2019
""""""

import numpy as np
import pprint
import pickle
from pyflwdir import (
    basin,
    basin_utils,
    basin_descriptors,
    core,
    core_d8,
    core_nextxy,
    core_nextidx,
    core_ldd,
    dem,
    gis_utils,
    subgrid,
    upscale,
)

ftypes = {
    core_d8._ftype: core_d8,
    core_ldd._ftype: core_ldd,
    core_nextxy._ftype: core_nextxy,
    core_nextidx._ftype: core_nextidx,
}
ftypes_str = " ,".join(list(ftypes.keys()))

# export
__all__ = ["FlwdirRaster", "from_array", "load"]


# import logging
# logger = logging.getLogger(__name__)

# TODO: this is slow on large arrays
def _infer_ftype(flwdir):
    """infer flowdir type from data"""
    ftype = None
    for _, fd in ftypes.items():
        if fd.isvalid(flwdir):
            ftype = fd._ftype
            break
    if ftype is None:
        raise ValueError("The flow direction type not recognized.")
    return ftype


def from_array(
    data, ftype="infer", check_ftype=True,
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
    if ftype == "infer":
        ftype = _infer_ftype(data)
        check_ftype = False  # already done
    if ftype == "nextxy":
        size = data[0].size
        shape = data[0].shape
    else:
        size = data.size
        shape = data.shape
    # TODO: check if this can go
    if len(shape) != 2:
        raise ValueError("Flow direction array should be 2 dimensional")

    # parse data
    fd = ftypes[ftype]
    if check_ftype and not fd.isvalid(data):
        raise ValueError(f'The flow direction type is not recognized as "{ftype}".')
    idxs_dense, idxs_ds, idxs_pit = fd.from_array(data)

    # initialize
    return FlwdirRaster(
        idxs_dense=idxs_dense,
        idxs_ds=idxs_ds,
        idxs_pit=idxs_pit,
        shape=shape,
        ftype=ftype,
    )


def load(fn):
    """Load serialized FlwdirRaster object from file
    
    Parameters
    ----------
    fn : str
        path
    """
    with open(fn, "rb") as handle:
        kwargs = pickle.load(handle)
    return FlwdirRaster(**kwargs)


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
        linear dense raster indices of pit cells
    isvalid : bool
        True if no loops in flow direction data    
    """

    def __init__(
        self, idxs_dense, idxs_ds, idxs_pit, shape, ftype,
    ):
        """Flow direction raster array parsed to actionable format.
        
        Parameters
        ----------
        data : ndarray
            flow direction raster data
        ftype : {'d8', 'ldd', 'nextxy', 'nextidx'}, optional
            name of flow direction type, infer from data if 'infer'
            (the default is 'infer')
        check_ftype : bool, optional
            check if valid flow direction raster, only if ftype is not 'infer'
            (the default is True)
        """
        if not ftype in ftypes.keys():
            raise ValueError(f"Unknown ftype {ftype}. Choose from {ftypes_str}")

        # properties
        self.shape = shape
        self.size = np.multiply(*self.shape)
        self.ftype = ftype
        self._core = ftypes[ftype]

        # data
        self._idxs_dense = idxs_dense
        self._idxs_ds = idxs_ds
        self._idxs_pit = idxs_pit
        self.ncells = self._idxs_dense.size

        # check validity
        if self.ncells < 1:
            raise ValueError("Invalid flow direction data: zero size ")
        elif self.ncells > 2 ** 32 - 2:  # maximum size we can have with uint32 indices
            raise ValueError(f"Too many active nodes, max: {2**32 - 2}.")
        if self._idxs_pit.size == 0:
            raise ValueError("Invalid flow direction data: no pits found")

        # set placeholder for upstream indices / network tree
        self._idxs_us_ = None  # 2D array with upstream indices
        self._tree_ = None  # List of array ordered from down- to upstream

    def __str__(self):
        return pprint.pformat(self._dict)

    @property
    def _dict(self):
        return {
            "ftype": self.ftype,
            "shape": self.shape,
            "idxs_dense": self._idxs_dense,
            "idxs_ds": self._idxs_ds,
            "idxs_pit": self._idxs_pit,
        }

    @property
    def pits(self):
        """Returns 1D raster indices of pits
        
        Returns
        -------
        1D array of int
            1D raster indices of pit
        """
        return self._idxs_dense[self._idxs_pit]

    def dump(self, fn):
        """Serialize object to file using pickle library.      
        """
        with open(fn, "wb") as handle:
            pickle.dump(self._dict, handle, protocol=-1)

    @property
    def _idxs_us(self):
        """internal property for 2D array of upstream indices"""
        if self._idxs_us_ is None:
            self._idxs_us_ = core._idxs_us(self._idxs_ds)
        return self._idxs_us_

    @property
    def _tree(self):
        """Returns the network tree: a list of arrays ordered from down- to 
        upstream. 
        
        Returns
        -------
        list of ndarray of int
            river network tree
        """
        if self._tree_ is None:
            self.set_tree()  # setup network
        return self._tree_

    @property
    def isvalid(self):
        """Returns True if the flow direction map is valid."""
        return core.loop_indices(self._idxs_ds, self._idxs_us).size == 0

    def set_pits(self, idxs_pit=None, streams=None):
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
            2D raster with cells flagged 'True' at stream cells, only used
            in combination with idxs_pit.
            (the default is None) 
        """
        if idxs_pit is None:
            idxs0 = core.pit_indices(self._idxs_ds)
        else:
            idxs_pit = np.asarray(idxs_pit).flatten()
            idxs0 = self._sparse_idx(idxs_pit)
            if streams is not None:  # snap to streams
                idxs0 = core.downstream_river(
                    idxs0, self._idxs_ds, self._sparsify(streams)
                )
        self._idxs_pit = idxs0
        self._tree_ = None  # reset network tree

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
        self._tree_ = core.network_tree(self._idxs_pit, self._idxs_us)

    def repair(self):
        """Repairs loops by set a pit at every cell witch does not drain to 
        a pit."""
        repair_idx = core.loop_indices(self._idxs_ds, self._idxs_us)
        if repair_idx.size > 0:
            # set pits for all loop indices !
            self._idxs_ds[repair_idx] = repair_idx
            self._idxs_pit = core.pit_indices[self._idxs_ds]

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
        return ftypes[ftype].to_array(self._idxs_dense, self._idxs_ds, self.shape)

    def basins(self, ids=None):
        """Returns a basin map with a unique IDs for every basin. 
        
        If IDs are not provided, these start from 1 and the background value is 0.
        
        Parameters
        ----------
        ids : ndarray of uint32, optional
            IDs of basins in some order as pits
            (by Default these are numbered from 1)

        Returns
        -------
        ndarray of uint32
            basin map
        """
        if ids is None:
            ids = np.arange(self._tree[0].size, dtype=np.uint32) + 1
        else:
            ids = np.asarray(ids).astype(np.uint32)
            if ids.size != self._tree[0].size:
                raise ValueError("ids size does not match number of pits")
            if np.any(ids == 0):
                raise ValueError("ids cannot contain a value zero")
        basids = basin.basins(self._tree, self._idxs_us, self._tree[0], ids)
        return self._densify(basids, nodata=np.uint32(0))

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
        idxs0 = self._sparse_idx(idxs)
        subbas = basin.basins(self._tree, self._idxs_us, idxs0)
        return self._densify(subbas, nodata=np.uint32(0))

    def pfafstetter(self, depth=1, uparea=None, min_upa=0.0):
        """Returns the pfafstetter coding for a single basin.
        
        Parameters
        ----------
        depth : int, optional
            Number of pfafsterrer layers
            (by default 1)
        uparea : 2D array of float, optional
            2D raster with upstream area 
            (by default None; calculated on the fly)
        min_upa : float, optional
            Minimum subbasin area
            (by default 0.0)
        
        Returns
        -------
        2D array of uint32
            subbasin map with pfafstetter coding
        """
        if self.pits.size > 1:
            msg = "Only implemented for a single basin, i.e. with one pit"
            raise NotImplementedError(msg)
        if uparea is None:
            uparea = self.upstream_area()
        elif not np.all(uparea.shape == self.shape):
            raise ValueError("'uparea' shape does not match with flow direction shape")
        pfaf = basin.pfafstetter(
            self._idxs_pit[0],
            self._tree,
            self._idxs_us,
            self._sparsify(uparea),
            min_upa=min_upa,
            depth=depth,
        )
        return self._densify(pfaf, np.uint32(0))

    def upstream_area(self, affine=gis_utils.IDENTITY, latlon=False, mult=1):
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
        mult : float, optional
            Multiplication factor for unit conversion
            (the default is 1)
        Returns
        -------
        2D array of float
            upstream area map [m2]
        """
        uparea = basin.upstream_area(
            self._tree,
            self._idxs_dense,
            self._idxs_us,
            self.shape[1],
            affine=affine,
            latlon=latlon,
        )
        return self._densify(uparea * mult, nodata=-9999.0)

    def stream_order(self):
        """Returns the Strahler Order map [1]_. 

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
        strord = basin.stream_order(self._tree, self._idxs_us)
        return self._densify(strord, nodata=np.int8(-1))

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
        2D array of material dtype
            accumulated material map
        """
        if not np.all(material.shape == self.shape):
            raise ValueError(
                "'Material' shape does not match with flow direction shape."
            )
        accu_flat = basin.accuflux(
            self._tree, self._idxs_us, self._sparsify(material), nodata
        )
        return self._densify(accu_flat, nodata=nodata)

    def vector(
        self, min_order=1, xs=None, ys=None, affine=gis_utils.IDENTITY, crs=None
    ):
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
            raise ImportError(
                "The `vector` method requires the additional "
                + "shapely and geopandas packages."
            )
        # get geoms and make geopandas dataframe
        if xs is None or ys is None:
            xs, ys = gis_utils.idxs_to_coords(self._idxs_dense, affine, self.shape)
        else:
            if xs.size != self.size or ys.size != self.size:
                raise ValueError(
                    "'xs' and/or 'ys' size does not match with  flow direction size"
                )
            xs, ys = xs.ravel()[self._idxs_dense], ys.ravel()[self._idxs_dense]
        geoms = gis_utils.idxs_to_geoms(self._idxs_ds, xs, ys)
        gdf = gp.GeoDataFrame(geometry=geoms, crs=crs)
        # get additional meta data
        gdf["stream_order"] = self.stream_order().flat[self._idxs_dense]
        gdf["pit"] = self._idxs_ds == np.arange(self.ncells)
        return gdf

    def upscale(self, scale_factor, method="com", uparea=None, **kwargs):
        """Upscale flow direction network to lower resolution. 
        Available methods are Connecting Outlets Method (COM) [2]_, 
        Effective Area Method (EAM) [3]_ and Double Maximum Method (DMM) [4]_.

        Note: This method only works for D8 or LDD flow directon data.
        
        .. [2] Eilander et al. in preperation (TODO update ref)
        .. [3] Yamazaki D, Masutomi Y, Oki T and Kanae S 2008 
          "An Improved Upscaling Method to Construct a Global River Map" APHW
        .. [4] Olivera F, Lear M S, Famiglietti J S and Asante K 2002 
          "Extracting low-resolution river networks from high-resolution digital 
          elevation models" Water Resour. Res. 38 13-1-13–8 
          Online: http://doi.wiley.com/10.1029/2001WR000726
    
        
        Parameters
        ----------
        scale_factor : int
            number gridcells in resulting upscaled gridcell
        method : {'com', 'eam', 'dmm'}
            upscaling method
            (by default 'com')
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
        methods = ["com", "eam", "dmm"]
        if self.ftype not in ["d8", "ldd"]:
            raise ValueError(
                "The upscale method only works for D8 or LDD flow directon data."
            )
        if method not in methods:
            methodstr = "', '".join(methods)
            raise ValueError(f"Unknown method, select from: '{methodstr}'")
        if uparea is None:
            uparea = self.upstream_area()
        elif not np.all(uparea.shape == self.shape):
            raise ValueError("'uparea' shape does not match with flow direction shape")
        # upscale flow directions
        fupscale = getattr(upscale, method)
        nextidx, subidxs_out = fupscale(
            subidxs_ds=self._idxs_ds,
            subidxs_dense=self._idxs_dense,
            subuparea=uparea.ravel(),  # NOTE: not sparse!
            subshape=self.shape,
            cellsize=scale_factor,
            **kwargs,
        )
        dir_lr = from_array(nextidx, ftype="nextidx", check_ftype=False)
        if not dir_lr.isvalid:
            raise ValueError(
                "The upscaled flow direction network is invalid. "
                + "Please provide a minimal reproducible example."
            )
        return dir_lr, subidxs_out

    def subarea(self, other, subidxs_out, affine=gis_utils.IDENTITY, latlon=False):
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
            raise ValueError(
                "invalid 'subidxs_out' with missing values at valid indices."
            )
        subare = subgrid.cell_area(
            self._sparse_idx(subidxs_out0),
            self._idxs_dense,
            self._idxs_us,
            self.shape,
            latlon=latlon,
            affine=affine,
        )
        return other._densify(subare, -9999.0)

    def subriver(
        self,
        other,
        subidxs_out,
        elevtn,
        uparea=None,
        min_uparea=0.0,
        latlon=False,
        affine=gis_utils.IDENTITY,
    ):
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
            raise ValueError("'elevtn' shape does not match with flow direction shape")
        if uparea is None:
            uparea = self.upstream_area(latlon=latlon, affine=affine)
        elif not np.all(uparea.shape == self.shape):
            raise ValueError("'uparea' shape does not match with flow direction shape")
        rivlen, rivslp = subgrid.river_params(
            subidxs_out=self._sparse_idx(subidxs_out0),
            subidxs_dense=self._idxs_dense,
            subidxs_ds=self._idxs_ds,
            subidxs_us=self._idxs_us,
            subuparea=uparea.flat[self._idxs_dense],
            subelevtn=elevtn.flat[self._idxs_dense],
            subshape=self.shape,
            min_uparea=min_uparea,
            latlon=latlon,
            affine=affine,
        )
        return other._densify(rivlen, -9999.0), other._densify(rivslp, -9999.0)

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
        subcon = subgrid.connected(
            self._sparse_idx(subidxs_out0), other._idxs_ds, self._idxs_ds
        )
        return other._densify(subcon, True)

    def adjust_elevation(self, elevtn):
        """Returns the hydrologically adjusted elevation where each downstream cell 
        has the same or lower elevation as the current cell.

        NOTE. elevation adjusted inplace!
        
        Parameters
        ----------
        elevnt : 2D array of float
            elevation raster
        
        Returns
        -------
        2D array of float
            elevation raster
        """
        if not np.all(elevtn.shape == self.shape):
            raise ValueError("'elevtn' shape does not match with flow direction shape")
        elevtn.flat[self._idxs_dense] = dem.adjust_elevation(
            idxs_ds=self._idxs_ds,
            idxs_us=self._idxs_us,
            tree=self._tree,
            elevtn_sparse=self._sparsify(elevtn),
        )
        return elevtn

    # def drainage_path_stats(self, rivlen, elevtn):
    #     if not np.all(rivlen.shape == self.shape):
    #         raise ValueError(
    #             "'rivlen' shape does not match with flow direction shape")
    #     if not np.all(elevtn.shape == self.shape):
    #         raise ValueError(
    #             "'elevtn' shape does not match with flow direction shape")
    #     df_out = basin_descriptors.mean_drainage_path_stats(
    #         self._tree, self._idxs_us, self._sparsify(rivlen),
    #         self._sparsify(elevtn))
    #     return df_out

    def _densify(self, data, nodata):
        """Return dense array from 1D sparse data, filled with nodata value."""
        return core._densify(data, self._idxs_dense, self.shape, nodata=nodata)

    def _sparsify(self, data):
        """Return sparse data array from dense data array."""
        return data.flat[self._idxs_dense]

    def _sparse_idx(self, idx):
        """Transform linear indices of dense array to sparse indices."""
        # NOTE: should we throw an error if any idx is invalid ?
        idx_sparse = core._sparse_idx(idx, self._idxs_dense, self.size)
        valid = np.logical_and(idx_sparse >= 0, idx_sparse < self.ncells)
        if not np.all(valid):
            raise IndexError("dense index outside valid cells")
        return idx_sparse


def _check_convert_subidxs_out(subidxs_out, other):
    """Convert 2D subidxs_out grid to 1D array at valid indices"""
    if not isinstance(other, FlwdirRaster):
        raise ValueError("'other' is not recognized as instance of FlwdirRaster")
    if not np.all(subidxs_out.shape == other.shape):
        raise ValueError(
            "'subidxs_out' shape does not match with `other` flow direction shape"
        )
    subidxs_out0 = subidxs_out.ravel()[other._idxs_dense]
    if np.any(subidxs_out0 == core._mv):
        raise ValueError("invalid 'subidxs_out' with missing values at valid indices.")
    return subidxs_out0

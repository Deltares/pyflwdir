# -*- coding: utf-8 -*-
""""""

import numpy as np
from affine import Affine
import pprint
import pickle
import logging
from pyflwdir import gis_utils as gis
from pyflwdir import (
    arithmetics,
    basins,
    core,
    core_d8,
    core_nextxy,
    core_ldd,
    dem,
    gis_utils,
    regions,
    unitcatchments,
    upscale,
    streams,
)

# global variables
FTYPES = {
    core_d8._ftype: core_d8,
    core_ldd._ftype: core_ldd,
    core_nextxy._ftype: core_nextxy,
}
AREA_FACTORS = {"m2": 1.0, "ha": 1e4, "km2": 1e6, "cell": 1}
_mv = core._mv

# export
__all__ = ["FlwdirRaster", "from_array", "load"]

# logging
logger = logging.getLogger(__name__)


def _infer_ftype(flwdir):
    """infer flowdir type from data"""
    ftype = None
    for _, fd in FTYPES.items():
        if fd.isvalid(flwdir):
            ftype = fd._ftype
            break
    if ftype is None:
        raise ValueError("The flow direction type could not be inferred.")
    return ftype


def from_array(
    data, ftype="infer", check_ftype=True, transform=gis.IDENTITY, latlon=False,
):
    """Flow direction raster array parsed to actionable format.
    
    Parameters
    ----------
    data : ndarray
        flow direction raster data
    ftype : {'d8', 'ldd', 'nextxy', 'infer'}, optional
        name of flow direction type, infer from data if 'infer', by default is 'infer'
    check_ftype : bool, optional
        check if valid flow direction raster if ftype is not 'infer', by default True
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping, by default using the 
        identity transform.
    latlon : bool, optional
        True if WGS84 coordinate reference system, by default False. If True it 
        converts the cell areas from degree to metres, otherwise it assumes cell areas 
        are in unit metres.

    """
    if ftype == "infer":
        ftype = _infer_ftype(data)
        check_ftype = False  # already done
    if ftype == "nextxy":
        shape = data[0].shape
        ndim = data[0].ndim
    else:
        ndim = data.ndim
        shape = data.shape

    # import pdb; pdb.set_trace()
    if ndim != 2:
        raise ValueError("The FlwdirRaster should be 2 dimensional")

    # parse data
    fd = FTYPES[ftype]
    if check_ftype and not fd.isvalid(data):
        raise ValueError(f'The flow direction type "{ftype}" is not recognized.')
    idxs_ds, idxs_pit, n = fd.from_array(data)

    # initialize
    return FlwdirRaster(
        idxs_ds=idxs_ds,
        idxs_pit=idxs_pit,
        shape=shape,
        ncells=n,
        ftype=ftype,
        transform=transform,
        latlon=latlon,
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
    """Flow direction raster array parsed to general actionable format."""

    def __init__(
        self,
        idxs_ds,
        shape,
        ftype,
        idxs_pit=None,
        idxs_seq=None,
        ncells=None,
        transform=gis.IDENTITY,
        latlon=False,
        cache=True,
    ):
        """Flow direction raster array 
        
        Parameters
        ----------
        idxs_ds : 1D-array of int
            linear index of next downstream cell
        shape : tuple of int
            shape of raster
        ftype : {'d8', 'ldd', 'nextxy'}
            name of flow direction type
        idxs_pit : ndarray of int, optional
            linear indices of pit
        idxs_pit : ndarray of int, optional
            linear indices of valid cells ordered from down- to upstream
        ncells : integer
            number of valid cells
        transform : affine transform
            Two dimensional affine transform for 2D linear mapping, by default using 
            the identity transform.
        latlon : bool, optional
            True if WGS84 coordinate reference system, by default False. If True it 
            converts the cell areas from degree to metres, otherwise it assumes cell 
            areas are in unit metres.
        
        """
        # flow direction type
        if not ftype in FTYPES.keys():
            ftypes_str = '" ,"'.join(list(FTYPES.keys()))
            raise ValueError(f'ftype "{ftype}" unknown')
        self.ftype = ftype
        self._core = FTYPES[ftype]

        # dimension and spatial properties
        self.size = idxs_ds.size
        if self.size <= 1:
            raise ValueError(f"Invalid FlwdirRaster: size {self.size}")
        if np.multiply(*shape) != self.size:
            msg = f"Invalid FlwdirRaster: shape {shape} does not match size {self.size}"
            raise ValueError(msg)
        self.shape = shape
        self.set_transform(transform, latlon)

        # data
        self.idxs_ds = idxs_ds
        self._pit = idxs_pit
        self._seq = idxs_seq
        self._ncells = ncells

        # set placeholders only used if cache if True
        self.cache = cache
        self._idxs_us_main = None

        # check validity
        if self.pits.size == 0:
            raise ValueError("Invalid FlwdirRaster: no pits found")

    def __str__(self):
        return pprint.pformat(self._dict)

    def __getitem__(self, idx):
        return self.idxs_ds[idx]

    ### PROPERTIES ###

    @property
    def _dict(self):
        return {
            "ftype": self.ftype,
            "shape": self.shape,
            "ncells": self._ncells,
            "transform": self.transform,
            "latlon": self.latlon,
            "idxs_ds": self.idxs_ds,
            "idxs_seq": self._seq,
            "idxs_pit": self._pit,
        }

    @property
    def idxs_us_main(self):
        """Returns indices of main upstream cell"""
        if self._idxs_us_main is None:
            idxs_us_main = self.main_upstream()
        else:
            idxs_us_main = self._idxs_us_main
        return idxs_us_main

    @property
    def idxs_seq(self):
        """Returns linear indices ordered from down- to upstream."""
        if self._seq is None:
            self.order_cells()
        return self._seq

    @property
    def ncells(self):
        """Returns number of valid cells."""
        if self._ncells is None:
            self._ncells = int(np.sum(self.rank >= 0))
        return self._ncells

    @property
    def rank(self):
        """Returns the rank, i.e. distance the outlet in no. of cells."""
        return core.rank(self.idxs_ds)[0].reshape(self.shape)

    @property
    def pits(self):
        """Returns linear indices of pits."""
        if self._pit is None:
            self._pit = core.pit_indices(self.idxs_ds)
        return self._pit

    @property
    def pit_coords(self):
        """Returns x, y coordinates of pits.
        
        Returns
        -------
        xs : ndarray of float
            x coordinates in coordinate reference system
        ys : ndarray of float
            y coordinates in coordinate reference system
        """
        return gis.idxs_to_coords(self.pits, self.transform, self.shape)

    @property
    def isvalid(self):
        """Returns True if the flow direction map is valid."""
        return np.all(self.rank != -1)

    @property
    def mask(self):
        """Returns boolean array of valid cells in flow direction raster."""
        return self.idxs_ds != _mv

    ### SET/MODIFY PROPERTIES ###

    def order_cells(self):
        """Order cells from down- to upstream."""
        rnk, n = core.rank(self.idxs_ds)
        self._seq = np.argsort(rnk)[-n:]
        self._ncells = n

    def main_upstream(self, uparea=None, cache=None):
        idxs_us_main = core.main_upstream(
            idxs_ds=self.idxs_ds, uparea=self._check_data(uparea, "uparea")
        )
        if cache or (cache is None and self.cache):
            self._idxs_us_main = idxs_us_main
        return idxs_us_main

    def add_pits(self, idx=None, xy=None, streams=None):
        """Add pits the flow direction raster.
        If `streams` is given, the pits are snapped to the first downstream True cell.
        
        Parameters
        ----------
        idx : array_like, optional
            linear indices of pits, by default is None.
        xy : tuple of array of float, optional
            x, y coordinates of pits, by default is None.
        streams : ndarray of bool, optional
            2D raster with cells flagged 'True' at stream cells, only used
            in combination with idx or xy, by default None. 
        """
        # coordinates to linear dense indices
        if (xy is not None and idx is not None) or (xy is None and idx is None):
            raise ValueError("Either idx or xy should be provided.")
        elif xy is not None:
            idx = gis.coords_to_idxs(*xy, self.transform, self.shape)
        idxs0 = np.atleast_1d(idx)
        # snap to streams
        streams = self._check_data(streams, "streams", optional=True)
        if streams is not None:
            idxs0 = self.snap(idxs0, streams)[0]
        # add pits
        self.idxs_ds[idxs0] = idxs0
        self._pit = np.unique(np.concatenate([self.pits, idxs0]))
        # reset order, ncells and upstream cell indices
        self._seq = None
        self._ncells = None
        self._idxs_us_main = None

    def set_transform(self, transform, latlon=False):
        """Set transform affine.
        
        Parameters
        ----------
        transform : affine transform
            Two dimensional affine transform for 2D linear mapping. The default is an 
            identity transform.
        latlon : bool, optional
            True if WGS84 coordinate reference system. If True it converts the cell 
            areas from degree to metres, otherwise it assumes cell areas are in unit 
            metres. The default is False.
        """
        if not isinstance(transform, Affine):
            try:
                transform = Affine(*transform)
            except TypeError:
                raise ValueError("Invalid transform.")
        self.transform = transform
        self.latlon = latlon

    def repair_loops(self):
        """Repair loops by setting a pit at every cell which does not drain to a pit."""
        repair_idx = core.loop_indices(self.idxs_ds)
        if repair_idx.size > 0:
            # set pits for all loop indices !
            self.add_pits(repair_idx)

    ### WRITE ###

    def dump(self, fn):
        """Serialize object to file using pickle library."""
        with open(fn, "wb") as handle:
            pickle.dump(self._dict, handle, protocol=-1)

    def to_array(self, ftype=None):
        """Return 2D flow direction raster. 
        
        Parameters
        ----------
        ftype : {'d8', 'ldd', 'nextxy'}, optional
            name of flow direction type, by default None; use input ftype.
        
        Returns
        -------
        2D array of int
            flow direction raster
        """
        if ftype is None:
            ftype = self.ftype
        if ftype in FTYPES:
            flwdir = FTYPES[ftype].to_array(self.idxs_ds, self.shape)
        else:
            raise ValueError(f'ftype "{ftype}" unknown')
        return flwdir

    ### LOCAL METHODS ###

    def path(self, idxs, mask=None, max_length=None, unit="cell", direction="down"):
        """Returns path of indices in down- or upstream direction until: 
        1) a pit is found (including) or now more upstream cells; or
        2) a True cell is found in mask (including); or
        3) the max_length threshold is exceeded. 
        
        Parameters
        ----------
        idxs : array_like of int
            linear index of start cells
        mask : ndarray of bool, optional
            True if stream cell.
        max_length : float, optional
            maximum length of trace
        unit : {'m', 'cell'}, optional
            unit of length, by default 'cell'
        direction : {'up', 'down'}, optional
            direction of path, be default 'down', i.e. downstream

        Returns
        -------
        list of 1D-array of int  
            linear indices of path
        1D-array of float
            distance along path between start and end cell
        """
        unit = str(unit).lower()
        if unit not in ["m", "cell"]:
            raise ValueError(f'Unknown unit: {unit}, select from ["m", "cell"].')
        direction = str(direction).lower()
        if direction == "up":
            idxs_nxt = self.idxs_us_main
        elif direction == "down":
            idxs_nxt = self.idxs_ds
        else:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        paths, dist = core.path(
            np.atleast_1d(idxs),
            idxs_nxt,
            mask=self._check_data(mask, "mask", optional=True),
            max_length=max_length,
            real_length=unit == "m",
            ncol=self.shape[1],
            latlon=self.latlon,
            transform=self.transform,
        )
        return paths, dist

    def snap(self, idxs, mask=None, max_length=None, unit="cell", direction="down"):
        """Returns the last index in down- or upstream direction where: 
        1) a pit is found (including) or now more upstream cells; or
        2) a True cell is found in mask (including)
        3) the max_length threshold is exceeded. 
        
        Parameters
        ----------
        idxs : array_like of int
            linear index of start cell.
        mask : ndarray of bool
            True at cell where to snap to
        max_length : float, optional
            maximum length of trace
        unit : {'m', 'cell'}, optional
            unit of length, by default 'cell'
        direction : {'up', 'down'}, optional
            direction of path, be default 'down', i.e. downstream

        Returns
        -------
        array_like of int  
            linear index of snapped cell
        array_like of float
            distance along path between start and snap cell.
        """
        unit = str(unit).lower()
        if unit not in ["m", "cell"]:
            raise ValueError(f'Unknown unit: {unit}, select from ["m", "cell"].')
        direction = str(direction).lower()
        if direction == "up":
            idxs_nxt = self.idxs_us_main
        elif direction == "down":
            idxs_nxt = self.idxs_ds
        else:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        idxs1, dist = core.snap(
            np.atleast_1d(idxs),
            idxs_nxt,
            mask=self._check_data(mask, "mask", optional=True),
            max_length=max_length,
            real_length=unit == "m",
            ncol=self.shape[1],
            latlon=self.latlon,
            transform=self.transform,
        )
        return idxs1, dist

    ### GLOBAL ARITHMETICS ###

    def downstream(self, data):
        """Returns next downstream value.
        
        Parameters
        ----------
        data : 2D array
            values
        
        Returns
        -------
        2D array
            downstream data
        """
        dflat = self._check_data(data, "data")
        return np.where(self.mask, dflat[self.idxs_ds], dflat).reshape(data.shape)

    def upstream_sum(self, data, mv=-9999):
        """Returns sum of next upstream values.
        
        Parameters
        ----------
        data : 2D array
            values
        mv : int or float
            missing value
        
        Returns
        -------
        2D array
            sum of upstream data
        """
        data_out = arithmetics.upstream_sum(
            idxs_ds=self.idxs_ds, data=self._check_data(data, "data"), nodata=mv
        )
        return data_out.reshape(data.shape)

    def moving_average(self, data, n, weights=None, nodata=-9999.0):
        """Take the moving weighted average over the flow direction network
        
        Parameters
        ----------
        data : 2D array
            values
        n : int
            number of up/downstream neighbors to include
        weights : 2D array, optional
            weights, by default equal weights are assumed
        nodata : float, optional
            Nodata values which is ignored when calculating the average, by default -9999.0

        Returns
        -------
        2D array
            averaged data
        """
        dflat = np.atleast_1d(data).ravel()
        if dflat.size != self.size:
            raise ValueError("Data size does not match with FlwdirRaster size.")
        if weights is None:
            weights = np.ones(self.size, dtype=np.float16)
        elif np.atleast_1d(weights).size != self.size:
            raise ValueError("Weights size does not match with FlwdirRaster size.")
        data_out = arithmetics.moving_average(
            data=dflat,
            weights=np.atleast_1d(weights).ravel(),
            n=n,
            idxs_ds=self.idxs_ds,
            idxs_us_main=self.idxs_us_main,
            nodata=nodata,
        )
        return data_out.reshape(data.shape)

    ### BASINS ###

    def basins(self, idxs=None, ids=None):
        """Returns a (sub)basin map with a unique ID for every (sub)basin. 
        To return a subbasin map linear indices of subbasin outlets should be provided. 
        By default, (sub)basin IDs start from 1 and the background value is 0.
        
        Parameters
        ----------
        idxs : ndarray of int
            linear indices of subbasin outlets
        ids : ndarray of uint32, optional
            IDs of (sub)basins in same order as idxs, by default None

        Returns
        -------
        2D array of uint32
            (sub)basin map
        """
        if idxs is None:  # full basins
            idxs = self.pits
        else:
            idxs = np.atleast_1d(idxs)
        if ids is not None:
            ids = np.atleast_1d(ids).ravel()
            if ids.size != idxs.size:
                raise ValueError("IDs size does not match size of idxs.")
            elif np.any(ids == 0):
                raise ValueError("IDs cannot contain a value zero.")
        basids = basins.basins(self.idxs_ds, idxs, self.idxs_seq, ids)
        return basids.reshape(self.shape)

    def basin_bounds(self, basins=None, **kwargs):
        """Returns a table with the basin boundaries. At index 0 the total bounding 
        box is given. 
        
        Additional key-word arguments are passed to the basins method which is used if 
        no basins map is provided.
        
        Parameters
        ----------
        basins : 2D array of uint32, optional
            basin ids, by default None and calculated on the fly.
            
        Returns
        -------
        pandas.DataFrame
            bounds of basins
        """
        if basins is None:
            basins = self.basins(**kwargs)
        elif basins.size != self.size:
            raise ValueError('"basins" size does not match with FlwdirRaster size')
        return regions.region_bounds(basins=basins, transform=self.transform)

    def pfafstetter(self, idx0, depth=1, uparea=None, upa_min=0.0):
        """Returns the pfafstetter coding for a single basin.
        
        Parameters
        ----------
        idx0 : int
            index of outlet cell
        depth : int, optional
            Number of pfafsterrer layers, by default 1.
        uparea : 2D array of float, optional
            2D raster with upstream area, by default None; calculated on the fly.
        upa_min : float, optional
            Minimum subbasin area, by default 0.0.
        
        Returns
        -------
        2D array of uint32
            subbasin map with pfafstetter coding
        """
        pfaf = basins.pfafstetter(
            idx0=idx0,
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            uparea=self._check_data(uparea, "uparea"),
            upa_min=upa_min,
            depth=depth,
        )
        return pfaf.reshape(self.shape)

    ### ACCUMULATE ####

    def upstream_area(self, unit="cell"):
        """Returns the upstream area map based on the flow direction map. 
        
        If latlon is True it converts the cell areas to metres, otherwise it
        assumes the coordinate unit is metres.
        
        Parameters
        ----------
        unit : {'m2', 'ha', 'km2', 'cell'}
            Upstream area unit.

        Returns
        -------
        2D array of float
            upstream area map [m2]
        """
        unit = str(unit).lower()
        if unit not in AREA_FACTORS:
            fstr = '", "'.join(AREA_FACTORS.keys())
            raise ValueError(f'Unknown unit: {unit}, select from "{fstr}".')
        uparea = streams.upstream_area(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            ncol=self.shape[1],
            latlon=False if unit == "cell" else self.latlon,
            transform=gis.IDENTITY if unit == "cell" else self.transform,
            area_factor=AREA_FACTORS[unit],
            nodata=np.int32(-9999) if unit == "cell" else np.float64(-9999),
            dtype=np.int32 if unit == "cell" else np.float64,
        )
        return uparea.reshape(self.shape)

    def accuflux(self, data, nodata=-9999):
        """Return accumulated data values along the flow direction map.
        

        Parameters
        ----------
        data : 2D array
            values
        nodata : int or float
            Missing data value for cells outside domain

        Returns
        -------
        2D array with data.dtype
            accumulated values
        """
        accu = streams.accuflux(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            data=self._check_data(data, "data"),
            nodata=nodata,
        )
        return accu.reshape(data.shape)

    ### STREAMS ####

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
        return streams.stream_order(self.idxs_ds, self.idxs_seq).reshape(self.shape)

    def stream_distance(self, mask=None, unit="cell"):
        """Returns distance to outlet or next downstream True cell in mask

        Parameters
        ----------
        mask : 1D-array of bool, optional
            True if stream cell
        unit : {'m', 'cell'}, optional
            length unit, by default 'cell'

        -------
        1D array of float
            distance to next downstream True cell, or outlet
        """
        unit = str(unit).lower()
        if unit not in ["m", "cell"]:
            raise ValueError(f'Unknown unit: {unit}, select from "m", "cell"')
        stream_dist = streams.stream_distance(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            ncol=self.shape[1],
            mask=self._check_data(mask, "mask", optional=True),
            real_length=unit != "cell",
            transform=self.transform,
            latlon=self.latlon,
        )
        return stream_dist.reshape(self.shape)

    def vectorize(self, mask=None, xs=None, ys=None, crs=None):
        """Returns a GeoDataFrame with river segments. Segments are LineStrings 
        connecting cell from down to upstream. 
        
        The coordinates of the nodes are based on the cell center as calculated 
        using the affine transform, unless maps with subgrid x and y 
        coordinates are provided.
        
        Parameters
        ----------
        min_order : int
            Minimum Strahler Order recognized as river, by the default 1.
        xs, ys : ndarray of float
            Raster with cell x, y coordinates, by default None and inferred from cell 
            center.
        crs : str, optional 
            Coordinate reference system used for the returned GeoDataFrame
        
        Returns
        -------
        geopandas.GeoDataFrame
            flow direction vector
        """
        # get geoms and make geopandas dataframe
        if xs is None or ys is None:
            idxs = np.arange(self.size, dtype=np.intp)
            xs, ys = gis.idxs_to_coords(idxs, self.transform, self.shape)
        df = gis.vectorize(
            idxs_ds=self.idxs_ds,
            xs=self._check_data(xs, "xs"),
            ys=self._check_data(ys, "ys"),
            mask=self._check_data(mask, "mask", optional=True),
            crs=crs,
        )
        return df

    ### UPSCALE FLOW DIRECTIONOS ###

    def upscale(self, scale_factor, method="com2", uparea=None, **kwargs):
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
        method : {'com2', 'com', 'eam', 'dmm'}
            upscaling method, by default 'com2'
        uparea : 2D array of float, optional
            2D raster with upstream area, by default None; calculated on the fly.

        Returns
        ------
        FlwdirRaster
            upscaled Flow Direction Raster
        ndarray of int
            1D raster indices of subgrid outlets
        """
        if self.ftype not in ["d8", "ldd"]:
            raise ValueError(
                "The upscale method only works for D8 or LDD flow directon data."
            )
        methods = ["com2", "com", "eam", "dmm"]
        method = str(method).lower()
        if method not in methods:
            methodstr = "', '".join(methods)
            raise ValueError(f"Unknown method: {method}, select from: '{methodstr}'")
        # upscale flow directions
        idxs_ds1, idxs_out, shape1 = getattr(upscale, method)(
            subidxs_ds=self.idxs_ds,
            subuparea=self._check_data(uparea, "uparea"),
            subshape=self.shape,
            cellsize=scale_factor,
            **kwargs,
        )
        transform1 = Affine(
            self.transform[0] * scale_factor,
            self.transform[1],
            self.transform[2],
            self.transform[3],
            self.transform[4] * scale_factor,
            self.transform[5],
        )
        # initialize new flwdir raster object
        flw1 = FlwdirRaster(
            idxs_ds=idxs_ds1,
            shape=shape1,
            transform=transform1,
            ftype=self.ftype,
            latlon=self.latlon,
        )
        if not flw1.isvalid:
            raise ValueError(
                "The upscaled flow direction network is invalid. "
                + "Please provide a minimal reproducible example."
            )
        return flw1, idxs_out.reshape(shape1)

    def upscale_connect(self, other, idxs_out):
        """Returns array with ones where the upscaled flow directions are connected 
        by stream paths in the highres (subgrid) flow direction map. Cells with missing 
        flow direction data have a value -1.
        
        Parameters
        ----------
        other : FlwdirRaster
            upscaled Flow Direction Raster
        idxs_out : 2D array of int
            linear indices of grid outlets
        
        Returns
        -------
        2D array of int8 with other.shape
            valid subgrid connection
        """
        subcon = upscale.connected(idxs_out.ravel(), other.idxs_ds, self.idxs_ds)
        return subcon.reshape(other.shape)

    ### UNIT CATCHMENT ###

    def ucat_outlets(self, cellsize, uparea=None, method="eam"):
        """Returns linear indices of unit catchment outlet cells.

        For more information about the methods see upscale script.

        Parameters
        ----------
        cellsize : int
            size of unit catchment measured in no. of higres cells
        uparea : 2D array of float, optional
            upstream area
        method : {"eam", "dmm"}, optional
            method to derive outlet cell indices, by default 'eam'
        
        Returns
        -------
        idxs_out : 2D array of int
            linear indices of unit catchment outlet cells
        """
        methods = ["eam", "dmm"]
        method = str(method).lower()
        if method not in methods:
            methodstr = "', '".join(methods)
            raise ValueError(f"Unknown method: {method}, select from: '{methodstr}'")
        idxs_out, shape1 = unitcatchments.outlets(
            idxs_ds=self.idxs_ds,
            uparea=self._check_data(uparea, "uparea"),
            cellsize=int(cellsize),
            shape=self.shape,
            method="eam",
        )
        return idxs_out.reshape(shape1)

    def ucat_area(self, idxs_out, unit="cell"):
        """Returns the unit catchment map (highres) and area (lowres) [m2]. 
        
        Parameters
        ----------
        idxs_out : 2D array of int
            linear indices of unit catchment outlets
        unit : {'m2', 'ha', 'km2', 'cell'}
            Upstream area unit.

        Returns
        -------
        2D array of float with other.shape
            subgrid cell area [m2]
        """
        unit = str(unit).lower()
        if unit not in AREA_FACTORS:
            fstr = '", "'.join(AREA_FACTORS.keys())
            raise ValueError(f'Unknown unit: {unit}, select from "{fstr}"')
        ucat_map, ucat_are = unitcatchments.area(
            idxs_out.ravel(),
            self.idxs_ds,
            self.idxs_seq,
            self.shape[1],
            transform=gis.IDENTITY if unit == "cell" else self.transform,
            latlon=False if unit == "cell" else self.latlon,
            area_factor=AREA_FACTORS[unit],
            nodata=np.int32(-9999) if unit == "cell" else np.float64(-9999),
            dtype=np.int32 if unit == "cell" else np.float64,
        )
        return ucat_map.reshape(self.shape), ucat_are.reshape(idxs_out.shape)

    def ucat_channel(self, idxs_out=None, elevtn=None, uparea=None, upa_min=0.0):
        """Returns the unit catchment river length [m] and slope [m/m] per lowres cell. 
        The unit catchment river is defined by the path starting at the unit catchment
        outlet cell moving upstream following the upstream subgrid cells with the 
        largest upstream area until it reaches the next upstream outlet cell. 
        
        A mimumum upstream area threshold can be set to discriminate river cells.
        
        Parameters
        ----------
        idxs_out : 2D array of int, optional
            linear indices of unit catchment outlets, if None (default) all valid 
            indices will be passed computing the cell length and slope in upstream
            direction.
        elevnt : 2D array of float, optional
            elevation raster, required to calculate slope
        uparea : 2D array of float, optional
            upstream area, if None (default) it is calculated.
        upa_min : float, optional
            Minimum upstream area to be consided as stream. Only used if 
            return_sublength is True, by default 0.0
        
        Returns
        -------
        2D array of float with other.shape
            subgrid river length [m]
        2D array of float with other.shape
            subgrid river slope [m/m]
        """
        if elevtn is None:
            elevtn = np.zeros(self.size, dtype=np.float16)
        if idxs_out is None:
            idxs_out = np.arange(self.size, dtype=np.intp).reshape(self.shape)
        rivlen, rivslp = unitcatchments.channel(
            idxs_out=idxs_out.ravel(),
            idxs_ds=self.idxs_ds,
            uparea=self._check_data(uparea, "uparea"),
            elevtn=self._check_data(elevtn, "elevtn"),
            ncol=self.shape[1],
            upa_min=upa_min,
            latlon=self.latlon,
            transform=self.transform,
        )
        return rivlen.reshape(idxs_out.shape), rivslp.reshape(idxs_out.shape)

    ### ELEVATION ###

    def dem_adjust(self, elevtn):
        """Returns the hydrologically adjusted elevation where each downstream cell 
        has the same or lower elevation as the current cell.
       
        Parameters
        ----------
        elevnt : 2D array of float
            elevation raster
        
        Returns
        -------
        2D array of float
            elevation raster
        """
        elevtn_out = dem.adjust_elevation(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            elevtn=self._check_data(elevtn, "elevtn"),
        )
        return elevtn_out.reshape(elevtn.shape)

    def hand(self, drain, elevtn):
        """Returns the height above the nearest drain (HAND), i.e.: the relative vertical 
        distance (drop) to the nearest dowstream river based on drainage‐normalized 
        topography and flowpaths. 

        Nobre A D et al. (2016) HAND contour: a new proxy predictor of inundation extent 
            Hydrol. Process. 30 320–33

        Parameters
        ----------
        drain : 2D array of bool
            drainage mask
        elevnt : 2D array of float
            elevation raster

        Returns
        -------
        2D array of float
            height above nearest drain
        """
        hand = dem.height_above_nearest_drain(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            drain=self._check_data(drain, "drain"),
            elevtn=self._check_data(elevtn, "elevtn"),
        )
        return hand.reshape(self.shape)

    def floodplains(self, drain, elevtn, uparea=None, b=0.3):
        """Returns floodplain boundaries based on a maximum treshold (h) of HAND which is 
        scaled with upstream area following h ~ A**b.  

        Nardi F et al (2019) GFPLAIN250m, a global high-resolution dataset of Earth’s 
            floodplains Sci. Data 6 180309

        Parameters
        ----------
        drain : 2D array of bool
            drainage mask
        elevnt : 2D array of float
            elevation raster [m]
        uparea : 2D array of float, optional
            upstream area raster [m2], by default calculated on the fly
        b : float, optional
            scale parameter, by default 0.3

        Returns
        -------
        1D array of int8
            floodplain 
        """
        fldpln = dem.floodplains(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            drain=self._check_data(drain, "drain"),
            elevtn=self._check_data(elevtn, "elevtn"),
            uparea=self._check_data(uparea, "uparea", unit="m2"),
            b=b,
        )
        return fldpln.reshape(self.shape)

    ### SHORTCUTS ###

    def _check_data(self, data, name, optional=False, **kwargs):
        """check or calculate upstream area cells; return flattened array"""
        if data is None and optional:
            return data
        elif data is None and name == "uparea":
            data = self.upstream_area(**kwargs)
        elif not np.atleast_1d(data).size == self.size:
            raise ValueError(f'"{name}" size does not match with FlwdirRaster size')
        return np.atleast_1d(data).ravel()

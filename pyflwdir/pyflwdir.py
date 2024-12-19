# -*- coding: utf-8 -*-
""""""

import numpy as np
from affine import Affine
import pickle
import logging
import warnings

from .flwdir import Flwdir
from . import gis_utils as gis
from . import (
    basins,
    core,
    core_d8,
    core_nextxy,
    core_ldd,
    dem,
    regions,
    subgrid,
    upscale,
    streams,
)

# global variables
FTYPES = {
    core_d8._ftype: core_d8,
    core_ldd._ftype: core_ldd,
    core_nextxy._ftype: core_nextxy,
}

# export
__all__ = ["FlwdirRaster", "from_array", "from_dem"]

# TODO logging
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


def from_dem(
    data,
    nodata=-9999.0,
    max_depth=-1.0,
    transform=gis.IDENTITY,
    latlon=False,
    outlets="edge",
):
    """Flow direction raster derived from digital elevation data based on steepest gradient.

    Outlets are assumed to only occur at the edge of valid elevation cells.
    Depressions elsewhere are filled based on its lowest pour point elevation.
    If the pour point depth is larger than the maximum pour point depth `max_depth` a pit
    is set at the depression local minimum elevation.

    Based on: Wang, L., & Liu, H. (2006). https://doi.org/10.1080/13658810500433453

    NOTE: to retrieve the depression filled dem, use the :py:func:`pyflwdir.dem.fill_depressions` method.

    Parameters
    ----------
    data : 2D array
        digital elevation data
    nodata : float, optional
        Missing data value, by default -9999.0
    max_depth: float, optional
        Maximum pour point depth. Depressions with a larger pour point
        depth are set as pit. A negative value (default) equals an infinitely
        large pour point depth causing all depressions to be filled.
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping, by default using the
        identity transform.
    latlon : bool, optional
        True if WGS84 coordinate reference system, by default False. If True it
        converts the cell areas from degree to metres, otherwise it assumes cell areas
        are in unit metres.
    outlets: {'edge', 'min'}
        Position for basin outlet(s) at the all valid elevation edge cell ('edge')
        or only the minimum elevation edge cell ('min')

    Returns
    -------
    FlwdirRaster
        Actionable flow direction object
    """
    # parse dem
    d8 = dem.fill_depressions(
        data, nodata=nodata, max_depth=max_depth, outlets=outlets
    )[1]
    return from_array(
        d8, ftype="d8", check_ftype=False, transform=transform, latlon=latlon
    )


def from_array(
    data,
    ftype="infer",
    check_ftype=True,
    mask=None,
    transform=gis.IDENTITY,
    latlon=False,
    **kwargs,
):
    """Flow direction raster array parsed to actionable format.

    Parameters
    ----------
    data : 2D array
        flow direction raster data
    ftype : {'d8', 'ldd', 'nextxy', 'infer'}, optional
        name of flow direction type, infer from data if 'infer', by default is 'infer'
    check_ftype : bool, optional
        check if valid flow direction raster if ftype is not 'infer', by default True
    mask : 2D array of bool, optional
        True for valid cells. Can be used to mask out subbasins.
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping, by default using the
        identity transform.
    latlon : bool, optional
        True if WGS84 coordinate reference system, by default False. If True it
        converts the cell areas from degree to metres, otherwise it assumes cell areas
        are in unit metres.
    **kwargs
        key-word arguments passed to FlwdirRaster

    Returns
    -------
    FlwdirRaster
        Actionable flow direction object

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
        raise ValueError(f'The flow direction data with type "{ftype}" is invalid.')
    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError('"mask" shape does not match with data shape')
        data = np.where(mask != 0, data, fd._mv)

    # use smallest possible dtype to represent indices
    n = data.size
    dtype = np.int32 if n < 2147483647 else (np.uint32 if n < 4294967294 else np.uint64)
    idxs_ds, idxs_pit, _ = fd.from_array(data, dtype=dtype)
    idxs_outlet = idxs_pit[np.isin(data.flat[idxs_pit], fd._pv)]

    # initialize
    return FlwdirRaster(
        idxs_ds=idxs_ds,
        idxs_pit=idxs_pit,
        idxs_outlet=idxs_outlet,
        shape=shape,
        ftype=ftype,
        transform=transform,
        latlon=latlon,
        **kwargs,
    )


class FlwdirRaster(Flwdir):
    """Flow direction raster array parsed to general actionable format."""

    def __init__(
        self,
        idxs_ds,
        shape,
        ftype,
        idxs_pit=None,
        idxs_outlet=None,
        idxs_seq=None,
        nnodes=None,
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
        idxs_pit, idxs_outlet : 2D array of int, optional
            linear indices of all pits/outlets,
            outlets exclude pits of incomplete basins at the domain boundary
        idxs_seq : 2D array of int, optional
            linear indices of valid cells ordered from down- to upstream
        nnodes : integer
            number of valid cells
        transform : affine transform
            Two dimensional affine transform for 2D linear mapping, by default using
            the identity transform.
        latlon : bool, optional
            True if WGS84 coordinate reference system, by default False. If True it
            converts the cell areas from degree to metres, otherwise it assumes cell
            areas are in unit metres.

        """
        # flow directions
        super().__init__(
            idxs_ds=idxs_ds,
            idxs_pit=idxs_pit,
            idxs_outlet=idxs_outlet,
            idxs_seq=idxs_seq,
            nnodes=nnodes,
            cache=cache,
        )

        # flow direction type
        if not ftype in FTYPES.keys():
            ftypes_str = '" ,"'.join(list(FTYPES.keys()))
            msg = f'Unknown flow direction type: "{ftype}", select from {ftypes_str}'
            raise ValueError(msg)
        self.ftype = ftype
        self._core = FTYPES[ftype]

        # raster dimensions and spatial attributes
        if np.multiply(*np.array(shape, np.uint64)) != self.size:
            msg = f"Invalid FlwdirRaster: shape {shape} does not match size {self.size}"
            raise ValueError(msg)
        self.shape = shape
        self.set_transform(transform, latlon)

    @property
    def _dict(self):
        return {
            "ftype": self.ftype,
            "shape": self.shape,
            "nnodes": self.nnodes,
            "transform": self.transform,
            "latlon": self.latlon,
            "idxs_ds": self.idxs_ds,
            "idxs_seq": self._seq,
            "idxs_pit": self._pit,
        }

    @property
    def ncells(self):
        return self.nnodes

    @property
    def idxs_seq(self):
        """Linear indices of valid cells ordered from down- to upstream."""
        if self._seq is None:
            self.order_cells(method="walk" if self.ftype != "nextxy" else "sort")
        return self._seq

    ### SET/MODIFY PROPERTIES ###

    def add_pits(self, idxs=None, xy=None, streams=None):
        """Add pits the flow direction raster.
        If `streams` is given, the pits are snapped to the first downstream True cell.

        Parameters
        ----------
        idxs : array_like, optional
            linear indices of pits, by default is None.
        xy : tuple of array_like of float, optional
            x, y coordinates of pits, by default is None.
        streams : 2D array of bool, optional
            2D raster with cells flagged 'True' at stream cells, only used
            in combination with idx or xy, by default None.
        """
        idxs1 = self._check_idxs_xy(idxs, xy, streams)
        super().add_pits(idxs=idxs1)

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

    ### WRITE / EXPORT ###

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
            flwdir = FTYPES[ftype].to_array(self.idxs_ds, self.shape, mv=self._mv)
        else:
            raise ValueError(f'ftype "{ftype}" unknown')
        return flwdir

    @staticmethod
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

    ### spatial methods ###

    def index(self, xs, ys, **kwargs):
        """Returns linear cell indices based on x, y coordinates.

        Parameters
        ----------
        xs, ys : ndarray of float
            x, y coordinates.

        Returns
        -------
        idxs : ndarray of int
            linear cell indices
        """
        return gis.coords_to_idxs(xs, ys, self.transform, self.shape, **kwargs)

    def xy(self, idxs, **kwargs):
        """Returns x, y coordinates of the cell center based on linear cell indices.

        Parameters
        ----------
        idxs : ndarray of int
            linear cell indices

        Returns
        -------
        xs : ndarray of float
            x coordinates.
        ys : ndarray of float
            y coordinates.
        """
        return gis.idxs_to_coords(idxs, self.transform, self.shape, **kwargs)

    @property
    def bounds(self):
        """Returns the raster bounding box [xmin, ymin, xmax, ymax]."""
        return np.array(gis.array_bounds(*self.shape, self.transform), dtype=np.float64)

    @property
    def extent(self):
        """Returns the raster extent in cartopy format [xmin, xmax, ymin, ymax]."""
        xmin, ymin, xmax, ymax = self.bounds
        return np.array([xmin, xmax, ymin, ymax], dtype=np.float64)

    @property
    def distnc(self):
        """Distance to outlet [m]"""
        if "distnc" in self._cached:
            distnc = self._cached["distnc"]
        else:
            distnc = self.stream_distance(unit="m")
            if self.cache:
                self._cached.update(distnc=distnc)
        return distnc

    @property
    def area(self):
        """Cell area [m]"""
        if "area" in self._cached:
            area = self._cached["area"]
        else:
            area = gis.area_grid(self.transform, self.shape, self.latlon, unit="m2")
            if self.cache:
                self._cached.update(area=area)
        return area

    ### LOCAL METHODS ###
    def path(
        self,
        idxs=None,
        xy=None,
        mask=None,
        max_length=None,
        unit="cell",
        direction="down",
    ):
        """Returns paths of indices in down- or upstream direction from the starting
        points until:
        1) a pit is found (including) or now more upstream cells are found; or
        2) a True cell is found in mask (including); or
        3) the max_length threshold is exceeded.

        To define starting points, either idxs or xy should be provided.

        Parameters
        ----------
        idxs : array_like, optional
            linear indices of starting point, by default is None.
        xy : tuple of array_like of float, optional
            x, y coordinates of starting point, by default is None.
        mask : 2D array of bool, optional
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
        if direction not in ["up", "down"]:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        paths, dist = core.path(
            idxs0=self._check_idxs_xy(idxs, xy),
            idxs_nxt=self.idxs_ds if direction == "down" else self.idxs_us_main,
            mask=self._check_data(mask, "mask", optional=True),
            max_length=max_length,
            real_length=unit == "m",
            ncol=self.shape[1],
            latlon=self.latlon,
            transform=self.transform,
            mv=self._mv,
        )
        return paths, dist

    def snap(
        self,
        idxs=None,
        xy=None,
        mask=None,
        max_length=None,
        unit="cell",
        direction="down",
    ):
        """Returns the last index in down- or upstream direction from the starting
        points where:

        1) a pit is found (including) or now more upstream cells are found; or
        2) a True cell is found in mask (including)
        3) the max_length threshold is exceeded.

        To define starting points, either idxs or xy should be provided.

        Parameters
        ----------
        idxs : array_like, optional
            linear indices of starting point, by default is None.
        xy : tuple of array_like of float, optional
            x, y coordinates of starting point, by default is None.
        mask : 2D array of bool
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
        if direction not in ["up", "down"]:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        idxs1, dist = core.snap(
            idxs0=self._check_idxs_xy(idxs, xy),
            idxs_nxt=self.idxs_ds if direction == "down" else self.idxs_us_main,
            mask=self._check_data(mask, "mask", optional=True),
            max_length=max_length,
            real_length=unit == "m",
            ncol=self.shape[1],
            latlon=self.latlon,
            transform=self.transform,
            mv=self._mv,
        )
        return idxs1, dist

    ### BASINS ###

    def basins(self, idxs=None, xy=None, ids=None, **kwargs):
        """Returns a (sub)basin map with a unique ID for every (sub)basin.

        To return a subbasin map either linear indices or x,y coordinates of subbasin
        outlets should be provided. Additional key-word arguments are passed to the
        snap method to snap the outlets to a downstream stream.

        By default, if IDs are not provided (sub)basin IDs start from 1. As the the
        background value of the basins map is zero, the IDs may not contain zeros.

        Parameters
        ----------
        idxs : array_like, optional
            linear indices of sub(basin) outlets, by default is None.
        xy : tuple of array_like of float, optional
            x, y coordinates of sub(basin) outlets, by default is None.
        ids : 1D array of uint32, optional
            IDs of (sub)basins in same order as idxs, by default None

        Returns
        -------
        2D array of uint32
            (sub)basin map
        """
        if idxs is None and xy is None:  # full basins / includes edge-pits
            idxs = self.idxs_pit
        else:
            idxs = self._check_idxs_xy(idxs, xy, **kwargs)
        if ids is not None:
            ids = np.atleast_1d(ids).ravel()
            if ids.size != idxs.size:
                raise ValueError("IDs size does not match size of idxs.")
            elif np.any(ids == 0):
                raise ValueError("IDs cannot contain a value zero.")
        basids = basins.basins(self.idxs_ds, idxs, self.idxs_seq, ids)
        return basids.reshape(self.shape)

    def subbasins_streamorder(self, strord=None, mask=None, min_sto=-2):
        """Returns a subbasin map with unique IDs and its outlet linear indices.
        Subbasins are defined based on a minimum stream order.

        Parameters
        ----------
        strord : 1D-array of uint8
            stream order
        mask : 2D array of bool
            Mask of valid cells.
        min_sto : int, optional
            minimum stream order of subbasins, by default the stream order is set to
            two under the global maximum stream order.

        Returns
        -------
        subbas : 2D-array of int32
            map with unique IDs for stream_order>=min_sto subbasins
        idxs_out: 1D array of int
            linear indices of subbasin outlet cells
        """
        subbas, idxs_out = basins.subbasins_streamorder(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            strord=self._check_data(strord, "strord"),
            mask=self._check_data(mask, "mask", optional=True),
            min_sto=min_sto,
        )
        return subbas.reshape(self.shape), idxs_out

    def subbasins_pfafstetter(self, depth=1, uparea=None, upa_min=0.0):
        """Returns the pfafstetter subbasins.

        Parameters
        ----------
        depth : int, optional
            Number of pfafstetter layers, by default 1.
        uparea : 2D array of float, optional
            2D raster with upstream area, by default None; calculated on the fly.
        upa_min : float, optional
            Minimum upstream area threshold for subbasins, by default 0.0.

        Returns
        -------
        subbas: 2D array of int32
            subbasin map with pfafstetter coding
        idxs_out: 1D array of int
            linear indices of subbasin outlet cells
        """
        uparea = self._check_data(uparea, "uparea")
        if upa_min is not None:
            mask = uparea >= upa_min
        subbas, idxs_out = basins.subbasins_pfafstetter(
            idxs_pit=self.idxs_pit,
            idxs_ds=self.idxs_ds,
            idxs_us_main=self.idxs_us_main,
            seq=self.idxs_seq,
            uparea=uparea,
            mask=mask,
            depth=depth,
            mv=self._mv,
        )
        return subbas.reshape(self.shape), idxs_out

    def subbasins_area(self, area_min, uparea=None):
        """Returns map with basin IDs, with a minimal area of `area_min`.
        Moving upstream from the basin outlets a new subbasin starts at tributaries
        with a contributing area larger than `area_min` and new interbasins when its area
        exceeds the `area_min`.

        Parameters
        ----------
        area_min : float
            subbasin area theshold; same unit as `uparea`, by default km2.
        uparea : 2D array of float, optional
            2D raster with upstream area, by default None; calculated on the fly.

        Returns
        -------
        subbas: 2D array of int32
            subbasin map with pfafstetter coding
        idxs_out: 1D array of int
            linear indices of subbasin outlet cells
        """
        subbas, idxs_out = basins.subbasins_area(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            idxs_us_main=self.idxs_us_main,
            uparea=self._check_data(uparea, "uparea", unit="km2"),
            area_min=area_min,
        )
        return subbas.reshape(self.shape), idxs_out

    def basin_bounds(self, basins=None, **kwargs):
        """Returns a the basin boundaries.

        Additional key-word arguments are passed to the basins method which is used to
        create a basins map if none is provided.

        Parameters
        ----------
        basins : 2D array of uint32, optional
            basin ids, by default None and calculated on the fly.

        Returns
        -------
        lbs : 1D array of int
            labels of basins
        bboxs : 2D array of float
            bounding boxes of basins, the columns represent [xmin, ymin, xmax, ymax]
        total_bbox : 1D array of float
            the total bounding box of all basins [xmin, ymin, xmax, ymax]
        """
        lbs, bboxs, total_bbox = regions.region_bounds(
            regions=self._check_data(basins, "basins", flatten=False, **kwargs),
            transform=self.transform,
        )
        return lbs, bboxs, total_bbox

    def basin_outlets(self, basins):
        """Returns the linear index of the outlet cell of `basins`.

        Parameters
        ----------
        basins: 2D array of int
            raster with unique IDs for each basin, where the background value must be zero.

        Returns
        -------
        lbs: 1D array
            array of the unique region IDs
        idxs_out: 1D array
            linear index of outlet cell per region
        """
        lbs, idxs_out = regions.region_outlets(
            regions=self._check_data(basins, "basins"),
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
        )
        return lbs, idxs_out

    def interbasin_mask(self, region, stream=None):
        """Returns most downstream contiguous area within region, i.e.: if a stream flows
        in and out of the region, only the most downstream contiguous area within region
        will be True in output mask. If a stream mask is provided the area is reduced to
        cells which drain to the stream.

        Parameters
        ----------
        region: 2D array of bool
            Initial mask of region
        stream: 2D array of bool
            True for stream cells

        Returns
        -------
        mask: 2D array of bool
            Mask of most downstream contiguous area within region
        """
        mask = basins.interbasin_mask(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            region=self._check_data(region, "region"),
            stream=self._check_data(stream, "stream", optional=True),
        )
        return mask.reshape(self.shape)

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
        if unit not in gis.AREA_FACTORS:
            fstr = '", "'.join(gis.AREA_FACTORS.keys())
            raise ValueError(f'Unknown unit: {unit}, select from "{fstr}".')
        if unit == "cell":
            area = np.ones(self.size, dtype=np.int32)
        else:
            area = self.area.ravel() / gis.AREA_FACTORS[unit]
        uparea = streams.accuflux(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            data=area,
            nodata=-9999,
        )
        uparea[~self.mask] = -9999
        return uparea.reshape(self.shape)

    ### STREAMS ####
    def inflow_idxs(self, region):
        """Returns linear indices of most upstream cells within region

        Parameters
        ----------
        region: 2D array of bool
            True where region
        Returns:
        -------
        idxs: 1D array of int
            linear indices
        """
        return core.inflow_idxs(
            self.idxs_ds, self.idxs_seq, self._check_data(region, "region")
        )

    def outflow_idxs(self, region):
        """Returns linear indices of most downstream cells within region

        Parameters
        ----------
        region: 2D array of bool
            True where region

        Returns:
        -------
        idxs: 1D array of int
            linear indices
        """
        return core.outflow_idxs(
            self.idxs_ds, self.idxs_seq, self._check_data(region, "region")
        )

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

    def vectorize(self, mask=None, xs=None, ys=None, direction="down", **kwargs):
        """Returns each flow direction as a linestring geo-feature

        Parameters
        ----------
        kind : {streams, flwdir}
            Kind of LineString features: either streams of local flow directions.
        mask : 2D array of bool
            Mask of valid cells.
        xs, ys : 2D array of float
            Raster with cell x, y coordinates, by default None and inferred from cell
            center.
        direction : {"up", "down"}
            Flow direction to define path between segment end points.
        kwargs : extra sample maps key-word arguments
            optional maps to sample from

        Returns
        -------
        feats : list of dict
            Geofeatures, to be parsed by e.g. geopandas.GeoDataFrame.from_features
        """
        idxs = core.flwdir_tuples(
            self.idxs_ds if direction == "down" else self.idxs_us_main,
            mask=self._check_data(mask, "mask", optional=True),
            mv=self._mv,
        )
        return self.geofeatures(idxs, xs=xs, ys=ys, **kwargs)

    def streams(
        self,
        mask=None,
        min_sto=1,
        xs=None,
        ys=None,
        idxs_out=None,
        max_len=0,
        direction="up",
        **kwargs,
    ):
        """Returns a list of stream segment as linestring geo-features.

        A stream segment is defined by flow path between two confluences
        or if `idxs_out` is given, two outlet cells by idxs_out. The stream cells
        can be set using a boolean `mask` or a minimum stream order `min_sto`.
        Note that if a mask is given the minimum stream order is ignored.

        Additional key-word arguments are maps from which a value is sampled
        at the most downstram cell of a stream segment.

        Parameters
        ----------
        kind : {streams, flwdir}
            Kind of LineString features: either streams of local flow directions.
        mask : 2D array of bool
            Mask of valid cells.
        min_sto : int
            Minimum Strahler Order recognized as river, by the default 1.
            A stream order map can optionally be passed using the key-word argument `strord`.
        xs, ys : 2D array of float
            Raster with cell x, y coordinates, by default None and inferred from cell
            center.
        idxs_out : 1D array of int
            Linear indices of segment end cells. Stream segments are based on  the path
            between two segment end cells in up- or downstream flow direction, see
            `direction` argument.
            By default None in which case segements are based on confluences.
        direction : {"up", "down"}
            Flow direction to define path between segment end points. Only used
            in combination with `idxs_out`.
        max_len: int, optional
            Maximum length of a single stream segment measured in cells.
            Longer streams segments are divided into smaller segments of equal length
            as close as possible to max_len.
        kwargs : extra sample maps key-word arguments
            optional maps to sample from

        Returns
        -------
        feats : list of dict
            Geofeatures, to be parsed by e.g. geopandas.GeoDataFrame.from_features
        """
        if mask is not None:
            mask = self._check_data(mask, "mask")
        elif min_sto > 1:
            strord = self._check_data(kwargs.get("strord"), "strord")
            mask = strord >= min_sto
            kwargs.update(strord=strord)  # add strord column

        if idxs_out is not None:
            idxs = subgrid.segment_indices(
                idxs_out=idxs_out,
                idxs_nxt=self.idxs_us_main if direction == "up" else self.idxs_ds,
                mask=mask,
                max_len=max_len,
                mv=self._mv,
            )
            # up to downstream for correct idx_ds column
            if direction == "up":
                idxs = [idxs0[::-1] for idxs0 in idxs]
        else:
            idxs = streams.streams(
                idxs_ds=self.idxs_ds,
                seq=self.idxs_seq,
                mask=mask,
                max_len=max_len,
                mv=self._mv,
            )

        return self.geofeatures(idxs, xs=xs, ys=ys, **kwargs)

    def geofeatures(self, flowpaths, xs=None, ys=None, **kwargs):
        """Returns a geo-features of flowpaths defined by a list of arrays of linear
        indices.

        The coordinates are based on the cell center as calculated
        using the affine transform, unless maps with (subgrid) x and y
        coordinates are provided.

        Parameters
        ----------
        flowpaths: list or 1D array of int
            list of flow paths described by linear indices
        xs, ys : 2D array of float
            Raster with cell x, y coordinates, by default None and inferred from cell
            center.
        kwargs : extra sample maps key-word arguments
            optional maps to sample from
            e.g.: strord=flw.stream_order()

        Returns
        -------
        feats : list of dict
            Geofeatures, to be parsed by e.g. geopandas.GeoDataFrame.from_features
        """
        # get geoms and return features
        feats = gis.features(
            flowpaths=flowpaths,
            xs=self._check_data(xs, "xs", optional=True),
            ys=self._check_data(ys, "ys", optional=True),
            transform=self.transform,
            shape=self.shape,
            **kwargs,
        )
        return feats

    ### UPSCALE FLOW DIRECTIONOS ###

    def upscale(self, scale_factor, method="ihu", uparea=None, **kwargs):
        """Upscale flow direction network to lower resolution.
        Available methods are Iterative hydrography upscaling method (IHU) [2]_,
        Effective Area Method (EAM) [3]_ and Double Maximum Method (DMM) [4]_.

        Note: This method only works for D8 or LDD flow directon data.

        .. [2] Eilander et al. in preperation (TODO update ref)
        .. [3] Yamazaki D, Masutomi Y, Oki T and Kanae S 2008
          "An Improved Upscaling Method to Construct a Global River Map" APHW
        .. [4] Olivera F, Lear M S, Famiglietti J S and Asante K 2002
          "Extracting low-resolution river networks from high-resolution digital
          elevation models" Water Resour. Res. 38 13-1-13-8
          Online: https://doi.org/10.1029/2001WR000726


        Parameters
        ----------
        scale_factor : int
            number gridcells in resulting upscaled gridcell
        method : {'ihu', 'eam_plus', 'eam', 'dmm'}
            upscaling method, by default 'ihu'
        uparea : 2D array of float or int, optional
            2D raster with upstream area, by default None; calculated on the fly.
        uparea : 2D array of int, optional
            2D raster with basin IDs, by default None. If provided it is used as an
            additional constrain to the IHU method to increase the upscaling speed.

        Returns
        ------
        flw: FlwdirRaster
            upscaled Flow Direction Raster
        idxs_out: 2D array of int
            linear indices of subgrid outlets
        """
        if self.ftype not in ["d8", "ldd"]:
            raise ValueError(
                "The upscale method only works for D8 or LDD flow directon data."
            )
        methods = ["ihu", "eam_plus", "com2", "com", "eam", "dmm"]
        method = str(method).lower()
        if method not in methods:
            methodstr = "', '".join(methods)
            raise ValueError(f"Unknown method: {method}, select from: '{methodstr}'")
        if "com" in method.lower():
            method_new = {"com": "eam_plus", "com2": "ihu"}.get(method.lower())
            warnings.warn(f"{method} renamed to {method_new}.", DeprecationWarning)
            method = method_new
        # upscale flow directions
        idxs_ds1, idxs_out, shape1 = getattr(upscale, method)(
            subidxs_ds=self.idxs_ds,
            subuparea=self._check_data(uparea, "uparea"),
            subshape=self.shape,
            cellsize=scale_factor,
            mv=self._mv,
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

    def upscale_error(self, other, idxs_out):
        """Returns an array with ones (True) where the upscaled flow directions are
        valid and zeros (False) where erroneous.

        The flow direction from cell 1 to cell 2 is valid if the first outlet pixel
        downstream of cell 1 is located in cell 2

        Cells with missing flow direction data have a value -1.

        Parameters
        ----------
        other : FlwdirRaster
            upscaled Flow Direction Raster
        idxs_out : 2D array of int
            linear indices of grid outlets

        Returns
        -------
        flwerr : 2D array of int8 with other.shape
            valid subgrid connection
        """
        assert self._mv == other._mv
        flwerr, _ = upscale.upscale_error(
            other._check_data(idxs_out, "idxs_out"),
            other.idxs_ds,
            self.idxs_ds,
            mv=self._mv,
        )
        return flwerr.reshape(other.shape)

    ### UNIT CATCHMENT ###

    def ucat_outlets(self, cellsize, uparea=None, method="eam_plus"):
        """Returns linear indices of unit catchment outlet pixel.

        For more information about the methods see upscale script.

        Parameters
        ----------
        cellsize : int
            size of unit catchment measured in no. of higres cells
        uparea : 2D array of float, optional
            upstream area
        method : {"eam_plus", "dmm"}, optional
            method to derive outlet cell indices, by default 'eam_plus'

        Returns
        -------
        idxs_out : 2D array of int
            linear indices of unit catchment outlet cells
        """
        methods = ["eam_plus", "dmm"]
        method = str(method).lower()
        if method not in methods:
            methodstr = "', '".join(methods)
            raise ValueError(f"Unknown method: {method}, select from: '{methodstr}'")
        idxs_out, shape1 = subgrid.outlets(
            idxs_ds=self.idxs_ds,
            uparea=self._check_data(uparea, "uparea"),
            cellsize=int(cellsize),
            shape=self.shape,
            method=method,
            mv=self._mv,
        )
        return idxs_out.reshape(shape1)

    def ucat_area(self, idxs_out, unit="cell"):
        """Returns the unit catchment map (highres) and area (lowres) [unit].

        Parameters
        ----------
        idxs_out : 2D array of int
            linear indices of unit catchment outlets
        unit : {'m2', 'ha', 'km2', 'cell'}
            area unit.

        Returns
        -------
        ucat_map: 2D array of int with self.shape
            unit catchment map [-]
        ucat_are: 2D array of float with idxs_out.shape
            subgrid cell area [unit]
        """
        unit = str(unit).lower()
        if unit not in gis.AREA_FACTORS:
            fstr = '", "'.join(gis.AREA_FACTORS.keys())
            raise ValueError(f'Unknown unit: {unit}, select from "{fstr}".')
        if unit == "cell":
            area = np.ones(self.size, dtype=np.int32)
        else:
            area = self.area.ravel() / gis.AREA_FACTORS[unit]
        ucat_map, ucat_are = subgrid.ucat_area(
            idxs_out=idxs_out.ravel(),
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            area=area,
            mv=self._mv,
        )
        return ucat_map.reshape(self.shape), ucat_are.reshape(idxs_out.shape)

    def ucat_volume(
        self, idxs_out, hand, depths=np.arange(0.5, 3.0, 0.5, dtype=np.float32)
    ):
        """Returns the unit catchment map (highres) and the
        flood volume at given flood depths (lowres) [m3].

        Parameters
        ----------
        idxs_out : 1D or 2D array of int
            linear indices of unit catchment outlets
        hand : 2D array of float
            Height Above Nearest Drain, see also :py:meth:`pyflwdir.FlwdirRaster.hand`
        depths : 1D array of float, optional
            Depth distribution of which to calculate the volume

        Returns
        -------
        ucat_map: 2D array of int with self.shape
            unit catchment map [-]
        ucat_vol: nD array of float with shape (depths.size, *idxs_out.shape)
            subgrid volume as function of depths [m3]
        """
        ucat_map, ucat_vol = subgrid.ucat_volume(
            idxs_out=idxs_out.ravel(),
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            area=self.area.ravel() / gis.AREA_FACTORS["m2"],
            hand=self._check_data(hand, "hand"),
            depths=depths,
            mv=self._mv,
        )
        shape_out = (depths.size, *idxs_out.shape)
        return ucat_map.reshape(self.shape), ucat_vol.reshape(shape_out)

    def subgrid_rivlen(
        self,
        idxs_out,
        mask=None,
        direction="up",
        unit="cell",
    ):
        """Returns the subgrid river length [m] based on unit catchment outlet locations.
        A cell's subgrid river is defined by the path starting at the unit
        catchment outlet pixel moving up- or downstream until it reaches the next
        outlet pixel. If moving upstream and a pixel has multiple upstream neighbors,
        the pixel with the largest upstream area is selected.

        Parameters
        ----------
        idxs_out : 2D array of int
            Linear indices of unit catchment outlets. If None (default) the cell
            size (instead of subgrid length) will be used.
        mask : 2D array of bool with self.shape, optional
            True for valid pixels. can be used to mask out pixels of small rivers.
        direction : {"up", "down"}
            Flow direction in which river length is measured, by default 'up'.
        unit : {'m', 'cell'}
            Upstream area unit.

        Returns
        -------
        rivlen : 2D array of float with idxs_out.shape
            subgrid river length [m]
        """
        direction = str(direction).lower()
        if direction not in ["up", "down"]:
            msg = f'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        if unit not in ["m", "cell"]:
            raise ValueError(f'Unknown unit: {unit}, select from ["m", "cell"]')
        if idxs_out is None:
            idxs_out = np.arange(self.size, dtype=np.intp).reshape(self.shape)
        distnc = self.distnc if unit == "m" else self.stream_distance(unit=unit)
        rivlen = subgrid.segment_length(
            idxs_out=idxs_out.ravel(),
            idxs_nxt=self.idxs_ds if direction == "down" else self.idxs_us_main,
            mask=self._check_data(mask, "mask", optional=True),
            distnc=distnc.ravel(),
            mv=self._mv,
        )
        shape = idxs_out.shape
        return rivlen.reshape(shape)

    def subgrid_rivslp(
        self,
        idxs_out,
        elevtn,
        length=1000,
        direction="both",
        method="mean",
        mask=None,
    ):
        """Returns the subgrid river slope [m/m] estimated at unit catchment outlet
        pixel. The slope is estimated from the elevation around the outlet pixel
        (`direction='both'`), or between the outlet pixel and the next downstream
        (`direction='down'`) or next upstream (`direction='up'`) outlet pixel.

        Parameters
        ----------
        idxs_out : 2D array of int
            Linear indices of unit catchment outlets, if None the cell
            size (instead of subgrid length) will be used.
        elevtn : 2D array of float with self.shape, optional
            Elevation raster, required to calculate slope.
        length : float, optional
            Subgrid river length [m] over which to calculate the slope, by default
            1000 m. Only used in combination with direction = 'both'
        direction : {"both", "up", "down"}
            Flow direction in which river slope is measured, by default 'both'.
        mask : 2D array of bool with self.shape, optional
            True for valid pixels. can be used to mask out pixels of small rivers.
        method: {'mean', 'lstsq'}
            Estimate the segment slope based on the `mean` slope: i.e.: net difference
            in elevation divided by length; or `lstsq` slope: i.e.: a simple ordinary
            least squares regression estimate of slope

        Returns
        -------
        rivslp : 2D array of float with idxs_out.shape
            subgrid river slope [m/m]
        """
        direction = str(direction).lower()
        if direction not in ["both", "up", "down"]:
            msg = f'Unknown flow direction: {direction}, select from ["both", "up", "down"].'
            raise ValueError(msg)
        if idxs_out is None:
            idxs_out = np.arange(self.size, dtype=np.intp).reshape(self.shape)
        if direction == "both":
            rivslp = subgrid.fixed_length_slope(
                idxs_out=idxs_out.ravel(),
                idxs_ds=self.idxs_ds,
                idxs_us_main=self.idxs_us_main,
                elevtn=self._check_data(elevtn, "elevtn"),
                distnc=self.distnc.ravel(),
                length=length,
                mask=self._check_data(mask, "mask", optional=True),
                mv=self._mv,
                lstsq=method == "lstsq",
            )
        else:
            rivslp = subgrid.segment_slope(
                idxs_out=idxs_out.ravel(),
                idxs_nxt=self.idxs_ds if direction == "down" else self.idxs_us_main,
                elevtn=self._check_data(elevtn, "elevtn"),
                distnc=self.distnc.ravel(),
                mask=self._check_data(mask, "mask", optional=True),
                mv=self._mv,
                lstsq=method == "lstsq",
            )
        return rivslp.reshape(idxs_out.shape)

    def subgrid_rivavg(
        self,
        idxs_out,
        data,
        weights=None,
        nodata=-9999.0,
        mask=None,
        direction="up",
    ):
        """Returns the average value over the subgrid river, based on unit catchment outlet
        locations. The subgrid river is defined by the path starting at the unit
        catchment outlet pixel moving up- or downstream until it reaches the next
        outlet pixel. If moving upstream and a pixel has multiple upstream neighbors,
        the pixel with the largest upstream area is selected.

        Parameters
        ----------
        idxs_out : 2D array of int
            Linear indices of unit catchment outlets. If None (default) the cell
            size (instead of subgrid length) will be used.
        data : 2D array
            values
        weigths : 2D array, optional
            weights used for averaging, by default None.
        nodata : int or float, optional
            Missing data value for cells outside domain, by default -9999.0
        mask : 2D array of bool with self.shape, optional
            True for valid pixels. can be used to mask out pixels of small rivers.
        direction : {"up", "down"}
            Flow direction in which segment is defined, by default 'up'.

        Returns
        -------
        rivavg : 2D array of float with idxs_out.shape
            subgrid segment average
        """
        direction = str(direction).lower()
        if direction not in ["up", "down"]:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        if idxs_out is None:
            idxs_out = np.arange(self.size, dtype=np.intp).reshape(self.shape)
        if weights is None:
            weights = np.ones(self.size, dtype=np.float32)
        rivavg = subgrid.segment_average(
            idxs_out=idxs_out.ravel(),
            idxs_nxt=self.idxs_ds if direction == "down" else self.idxs_us_main,
            data=self._check_data(data, "data"),
            weights=weights,
            nodata=nodata,
            mask=self._check_data(mask, "mask", optional=True),
            mv=self._mv,
        )
        shape = idxs_out.shape
        return rivavg.reshape(shape)

    def subgrid_rivmed(
        self,
        idxs_out,
        data,
        weights=None,
        nodata=-9999.0,
        mask=None,
        direction="up",
    ):
        """Returns the median value over the subgrid river, based on unit catchment outlet
        locations. The subgrid river is defined by the path starting at the unit
        catchment outlet pixel moving up- or downstream until it reaches the next
        outlet pixel. If moving upstream and a pixel has multiple upstream neighbors,
        the pixel with the largest upstream area is selected.

        Parameters
        ----------
        idxs_out : 2D array of int
            Linear indices of unit catchment outlets. If None (default) the cell
            size (instead of subgrid length) will be used.
        data : 2D array
            values
        weigths : 2D array, optional
            weights used for averaging, by default None.
        nodata : int or float, optional
            Missing data value for cells outside domain, by default -9999.0
        mask : 2D array of bool with self.shape, optional
            True for valid pixels. can be used to mask out pixels of small rivers.
        direction : {"up", "down"}
            Flow direction in which river length is measured, by default 'up'.

        Returns
        -------
        rivmed : 2D array of float with idxs_out.shape
            subgrid segment median
        """
        direction = str(direction).lower()
        if direction not in ["up", "down"]:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        if idxs_out is None:
            idxs_out = np.arange(self.size, dtype=np.intp).reshape(self.shape)
        if weights is None:
            weights = np.ones(self.size, dtype=np.float32)
        rivmed = subgrid.segment_median(
            idxs_out=idxs_out.ravel(),
            idxs_nxt=self.idxs_ds if direction == "down" else self.idxs_us_main,
            data=self._check_data(data, "data"),
            weights=weights,
            nodata=nodata,
            mask=self._check_data(mask, "mask", optional=True),
            mv=self._mv,
        )
        shape = idxs_out.shape
        return rivmed.reshape(shape)

    ### ELEVATION ###

    def dem_dig_d4(self, elevtn, rivmsk=None, nodata=-9999.0):
        """Returns the hydrologically adjusted elevation where for
        each cell river cell there is an adjacent D4 connected cell which has
        has the same or lower elevation as the current cell.

        Parameters
        ----------
        elevtn : 2D array of float
            elevation raster
        rivmsk : 2D array of bool, optional
            river mask

        Returns
        -------
        elv_out: 2D array of float
            elevation raster
        """
        elv_out = dem.dig_4connectivity(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            elv_flat=self._check_data(elevtn, "elevtn"),
            mask=self._check_data(rivmsk, "rivmsk", optional=True),
            shape=self.shape,
            nodata=nodata,
        )
        return elv_out.reshape(self.shape)

    def hand(self, drain, elevtn):
        """Returns the height above the nearest drain (HAND), i.e.: the relative vertical
        distance (drop) to the nearest dowstream river based on drainage-normalized
        topography and flowpaths.

        Nobre A D et al. (2016) HAND contour: a new proxy predictor of inundation extent
            Hydrol. Process. 30 320-33

        Parameters
        ----------
        drain : 2D array of bool
            drainage mask
        elevtn : 2D array of float
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

    def floodplains(self, elevtn, uparea=None, upa_min=1000, b=0.3):
        """Returns floodplain boundaries based on a maximum treshold (h) of HAND which is
        scaled with upstream area (A) following h ~ A**b.

        Nardi F et al (2019) GFPLAIN250m, a global high-resolution dataset of Earth's
            floodplains Sci. Data 6 180309

        Parameters
        ----------
        elevtn : 2D array of float
            elevation raster [m]
        uparea : 2D array of float, optional
            upstream area raster [km2], by default calculated on the fly
        b : float, optional
            scale parameter, by default 0.3
        upa_min : float, optional
            minimum upstream area threshold for streams [km2].

        Returns
        -------
        1D array of int8
            floodplain
        """
        fldpln = dem.floodplains(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            elevtn=self._check_data(elevtn, "elevtn"),
            uparea=self._check_data(uparea, "uparea", unit="km2"),
            upa_min=upa_min,
            b=b,
        )
        return fldpln.reshape(self.shape)

    ### SHORTCUTS ###

    def _check_data(self, data, name, optional=False, flatten=True, **kwargs):
        """check or calculate upstream area cells; return flattened array"""
        if data is None and optional:
            return
        if data is None:
            if name == "uparea":
                data = self.upstream_area(**kwargs)
            elif name == "basins":
                data = self.basins(**kwargs)
            elif name == "strord":
                data = self.stream_order(**kwargs)
        return super()._check_data(data, name, optional, flatten=flatten)

    def _check_idxs_xy(self, idxs=None, xy=None, streams=None):
        if (xy is not None and idxs is not None) or (xy is None and idxs is None):
            raise ValueError("Either idxs or xy should be provided.")
        elif xy is not None:
            idxs = self.index(*xy)
        return super()._check_idxs_xy(idxs, streams)

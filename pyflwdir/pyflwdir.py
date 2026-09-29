"""Main flow direction raster class and methods."""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Literal, cast, overload

import numpy as np
from affine import Affine

from . import (
    basins,
    core,
    core_d8,
    core_ldd,
    core_nextxy,
    dem,
    regions,
    streams,
    subgrid,
    upscale,
)
from . import gis_utils as gis
from .flwdir import Flwdir

# global variables
FTYPES = {
    core_d8._ftype: core_d8,
    core_ldd._ftype: core_ldd,
    core_nextxy._ftype: core_nextxy,
}

# export
__all__ = ["FlwdirRaster", "from_array", "from_dem"]

logger = logging.getLogger(__name__)


def _infer_ftype(flwdir: np.ndarray) -> Literal["d8", "ldd", "nextxy"]:
    """infer flowdir type from data"""
    ftype = None
    for fd in FTYPES.values():
        if fd.isvalid(flwdir):
            ftype = fd._ftype
            break
    if ftype is None:
        raise ValueError("The flow direction type could not be inferred.")
    return ftype


def from_dem(
    data: np.ndarray,
    nodata: float = -9999.0,
    max_depth: float = -1.0,
    transform: Affine = gis.IDENTITY,
    latlon: bool = False,
    outlets: Literal["edge", "min"] = "edge",
) -> FlwdirRaster:
    """Derive local D8 flow directions from digital elevation data.

    Outlets are assumed to only occur at the edge of valid elevation cells.
    Depressions elsewhere are filled to their lowest pour-point elevation. If the pour
    point depth is greater than or equal to `max_depth`, a pit is set at the depression's
    local minimum elevation.

    Based on: Wang, L., & Liu, H. (2006). https://doi.org/10.1080/13658810500433453

    NOTE: to retrieve the depression filled dem, use the :py:func:`pyflwdir.dem.fill_depressions` method.

    Parameters
    ----------
    data : 2D array
        digital elevation data
    nodata : float, optional
        Missing data value, by default -9999.0
    max_depth : float, optional
        Maximum pour point depth. Depressions with a larger pour point
        depth are set as pits. A negative value (default) represents an infinitely
        large pour point depth causing all depressions to be filled.
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping, by default using the
        identity transform.
    latlon : bool, optional
        True if WGS84 coordinate reference system, by default False. If True it
        converts the cell areas from degree to metres, otherwise it assumes cell areas
        are in unit metres.
    outlets : {'edge', 'min'}, optional
        Place outlets at all valid edge cells ('edge', default) or only at the
        lowest-elevation valid edge cell ('min').

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


def _get_idxs_dtype(n: int) -> type:
    """Return the smallest integer dtype that can represent ``n`` indices.

    A signed ``int64`` (rather than ``uint64``) is used for the largest
    rasters: mixing unsigned 64-bit indices with the signed missing value
    (``core._mv``) or signed integers promotes them to ``float64`` in numba,
    which then fails to index the arrays (see #79). ``int64`` still covers up
    to ``2**63 - 1`` cells, far beyond any realistic raster size.

    Parameters
    ----------
    n : int
        Number of indices (i.e. raster cells) to represent.

    Returns
    -------
    dtype : numpy data type
    """
    if n < 2147483647:  # 2**31 - 1
        return np.int32
    elif n < 4294967294:  # 2**32 - 2
        return np.uint32
    return np.int64


def from_array(
    data: np.ndarray,
    ftype: Literal["d8", "ldd", "nextxy", "infer"] = "infer",
    check_ftype: bool = True,
    mask: np.ndarray | None = None,
    transform: Affine = gis.IDENTITY,
    latlon: bool = False,
    **kwargs,
) -> FlwdirRaster:
    """Parse a flow-direction raster into the actionable `FlwDirRaster` format.

    Parameters
    ----------
    data : 2D array
        2D flow-direction raster. For `ftype='nextxy'`, provide the pair of row and
        column arrays.
    ftype : {'d8', 'ldd', 'nextxy', 'infer'}, optional
        Flow-direction convention. Use 'infer' to detect it from `data`, by default
        'infer'.
    check_ftype : bool, optional
        Whether to validate `data` against `ftype`, by default True. Validation is
        skipped when `ftype='infer'` because inference performs the check.
    mask : 2D array of bool, optional
        True for valid cells. Can be used to exclude cells outside the domain.
    transform : affine transform
        Affine transform mapping pixel coordinates to map coordinates, by default the
        identity transform.
    latlon : bool, optional
        Whether coordinates use the WGS84 geographic coordinate system. If True, cell
        areas are converted from degrees to square metres; otherwise coordinates are
        assumed to use metres, by default False.
    **kwargs : dict
        Additional keyword arguments passed to the `FlwDirRaster` constructor, such as
        `cache`.

    Returns
    -------
    FlwdirRaster
        Parsed flow-direction raster.

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
    dtype = _get_idxs_dtype(shape[0] * shape[1])
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
    """Flow-direction raster parsed into a common actionable format."""

    def __init__(
        self,
        idxs_ds: np.ndarray,
        shape: tuple,
        ftype: Literal["d8", "ldd", "nextxy"],
        idxs_pit: np.ndarray | None = None,
        idxs_outlet: np.ndarray | None = None,
        idxs_seq: np.ndarray | None = None,
        nnodes: int | None = None,
        transform: Affine = gis.IDENTITY,
        latlon: bool = False,
        cache: bool = True,
    ):
        """Initialize a flow-direction raster from downstream-cell indices.

        Parameters
        ----------
        idxs_ds : 1D-array of int
            Linear index of the next downstream cell for each raster cell.
        shape : tuple of int
            Raster dimensions as `(height, width)`.
        ftype : {'d8', 'ldd', 'nextxy'}
            Flow-direction convention.
        idxs_pit, idxs_outlet : np.ndarray of int, optional
            Indices of pit or outlet cells. `idxs_outlet` excludes pits of incomplete
            basins at the domain boundary.
        idxs_seq : np.ndarray of int, optional
            Valid cell indices ordered from downstream to upstream.
        nnodes : int, optional
            Number of valid cells. Calculated when needed if omitted.
        transform : Affine, optional
            Affine transform mapping pixel coordinates to map coordinates, by default
            the identity transform.
        latlon : bool, optional
            Whether coordinates use the WGS84 geographic coordinate system. If True,
            cell areas are converted from degrees to square metres; otherwise
            coordinates are assumed to use metres, by default False.
        cache : bool, optional
            Whether to cache derived arrays, by default True.

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
        if ftype not in FTYPES:
            ftypes_str = '", "'.join(list(FTYPES.keys()))
            msg = f'Unknown flow direction type: "{ftype}", select from "{ftypes_str}"'
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
    def _dict(self) -> dict:
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
    def ncells(self) -> int:
        """Number of valid cells in the flow-direction raster."""
        return self.nnodes

    @property
    def idxs_seq(self) -> np.ndarray:
        """Linear indices of valid cells ordered from down- to upstream."""
        if self._seq is None:
            self.order_cells(method="walk")
        return cast(np.ndarray, self._seq)

    ### SET/MODIFY PROPERTIES ###

    def add_pits(  # type: ignore[override]
        self,
        idxs: np.ndarray | None = None,
        xy: tuple[np.ndarray, np.ndarray] | None = None,
        streams: np.ndarray | None = None,
    ) -> None:
        """Add pits to the flow-direction raster.

        If `streams` is given, each pit is snapped to the first downstream True cell.

        Parameters
        ----------
        idxs : array_like, optional
            Linear indices of pit cells.
        xy : tuple of np.ndarray of float, optional
            x and y coordinates of pit cells.
        streams : np.ndarray of bool, optional
            Boolean raster marking stream cells. When provided, pit locations are
            snapped to the first downstream True cell.
        """
        idxs1 = self._check_idxs_xy(idxs, xy, streams)
        super().add_pits(idxs=idxs1)

    def set_transform(self, transform: Affine, latlon: bool = False) -> None:
        """Set the affine transform and coordinate-system type.

        Parameters
        ----------
        transform : affine transform
            Affine transform mapping pixel coordinates to map coordinates.
        latlon : bool, optional
            Whether coordinates use the WGS84 geographic coordinate system. If True,
            cell areas are converted from degrees to square metres; otherwise
            coordinates are assumed to use metres, by default False.
        """
        if not isinstance(transform, Affine):
            try:
                transform = Affine(*transform)
            except TypeError:
                raise ValueError("Invalid transform.")
        self.transform = transform
        self.latlon = latlon
        for key in ("area", "distnc", "idxs_us_main"):
            self._cached.pop(key, None)

    ### WRITE / EXPORT ###

    def to_array(
        self, ftype: Literal["d8", "ldd", "nextxy"] | None = None
    ) -> np.ndarray:
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
    def load(fn: str | Path) -> FlwdirRaster:
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

    def index(self, xs: np.ndarray, ys: np.ndarray, **kwargs) -> np.ndarray:
        """Returns linear cell indices based on x, y coordinates.

        Parameters
        ----------
        xs, ys : ndarray of float
            x, y coordinates.
        **kwargs : dict
            Additional coordinate-conversion options passed to `gis.coords_to_idxs`,
            such as `op` and `precision`.

        Returns
        -------
        idxs : ndarray of int
            linear cell indices
        """
        return gis.coords_to_idxs(xs, ys, self.transform, self.shape, **kwargs)

    def xy(self, idxs: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Returns x, y coordinates of the cell center based on linear cell indices.

        Parameters
        ----------
        idxs : ndarray of int
            linear cell indices
        **kwargs : dict
            Additional coordinate-conversion options passed to `gis.idxs_to_coords`,
            such as `offset`.

        Returns
        -------
        xs : ndarray of float
            x coordinates.
        ys : ndarray of float
            y coordinates.
        """
        return gis.idxs_to_coords(idxs, self.transform, self.shape, **kwargs)

    @property
    def bounds(self) -> np.ndarray:
        """Returns the raster bounding box [xmin, ymin, xmax, ymax]."""
        nrow, ncol = self.shape
        return np.array(gis.array_bounds(nrow, ncol, self.transform), dtype=np.float64)

    @property
    def extent(self) -> np.ndarray:
        """Returns the raster extent in cartopy format [xmin, xmax, ymin, ymax]."""
        xmin, ymin, xmax, ymax = self.bounds
        return np.array([xmin, xmax, ymin, ymax], dtype=np.float64)

    @property
    def distnc(self) -> np.ndarray:
        """Distance to outlet [m]"""
        if "distnc" in self._cached:
            distnc = self._cached["distnc"]
        else:
            distnc = self.stream_distance(unit="m")
            if self.cache:
                self._cached.update(distnc=distnc)
        return distnc

    @property
    def area(self) -> np.ndarray:
        """Cell area [m2]."""
        if "area" in self._cached:
            area = self._cached["area"]
        else:
            area = gis.area_grid(self.transform, self.shape, self.latlon, unit="m2")
            if self.cache:
                self._cached.update(area=area)
        return area

    ### LOCAL METHODS ###
    def path(  # type: ignore[override]
        self,
        idxs: np.ndarray | None = None,
        xy: tuple[np.ndarray, np.ndarray] | None = None,
        mask: np.ndarray | None = None,
        max_length: float | None = None,
        unit: Literal["m", "cell"] = "cell",
        direction: Literal["up", "down"] = "down",
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Trace paths downstream or upstream from starting cells.

        A path ends at a pit, at a True cell in `mask`, or when `max_length` is
        exceeded. The endpoint is included in the returned path. Provide either `idxs`
        or `xy` to specify starting cells.

        Parameters
        ----------
        idxs : array_like, optional
            Linear indices of starting cells.
        xy : tuple of array_like of float, optional
            x and y coordinates of starting cells.
        mask : 2D array of bool, optional
            True for cells where tracing stops; the matching cell is included.
        max_length : float, optional
            Maximum path length, measured in `unit`.
        unit : {'m', 'cell'}, optional
            Length unit, either metres ('m') or cells ('cell'), by default 'cell'.
        direction : {'up', 'down'}, optional
            Trace downstream ('down', default) or upstream ('up').

        Returns
        -------
        list of 1D-array of int
            List of arrays containing the linear indices in each path.
        1D array of float
            Distance from each start cell to its path endpoint, in `unit`.
        """
        if unit not in ["m", "cell"]:
            raise ValueError(f'Unknown unit: {unit}, select from ["m", "cell"].')
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
            transform=np.asarray(self.transform),
            mv=self._mv,
        )
        return paths, dist

    def snap(
        self,
        idxs: np.ndarray | None = None,
        xy: tuple[np.ndarray, np.ndarray] | None = None,
        mask: np.ndarray | None = None,
        max_length: float | None = None,
        unit: Literal["m", "cell"] = "cell",
        direction: Literal["up", "down"] = "down",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Snap starting cells to a downstream or upstream target.

        Tracing stops at a pit, at a True cell in `mask`, or when `max_length` is
        exceeded. Provide either `idxs` or `xy` to specify starting cells.

        Parameters
        ----------
        idxs : array_like, optional
            Linear indices of starting cells.
        xy : tuple of array_like of float, optional
            x and y coordinates of starting cells.
        mask : 2D array of bool
            True for target cells. The first target encountered is returned.
        max_length : float, optional
            Maximum tracing distance, measured in `unit`.
        unit : {'m', 'cell'}, optional
            Distance unit, either metres ('m') or cells ('cell'), by default 'cell'.
        direction : {'up', 'down'}, optional
            Trace downstream ('down', default) or upstream ('up').

        Returns
        -------
        array_like of int
            Linear index of the snapped cell for each starting cell.
        1D array of float
            Distance from each starting cell to its snapped cell, in `unit`.
        """
        if unit not in ["m", "cell"]:
            raise ValueError(f'Unknown unit: {unit}, select from ["m", "cell"].')
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
            transform=np.asarray(self.transform),
            mv=self._mv,
        )
        return idxs1, dist

    ### BASINS ###

    def basins(
        self,
        idxs: np.ndarray | None = None,
        xy: tuple[np.ndarray, np.ndarray] | None = None,
        ids: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Return a basin map with a unique ID for each basin.

        Provide outlet indices or coordinates to delineate subbasins. Additional
        keyword arguments are passed to `snap()` to snap outlet locations to a
        downstream stream.

        If `ids` is omitted, basin IDs start at 1. Zero is reserved for background and
        cannot be used as a basin ID.

        Parameters
        ----------
        idxs : array_like, optional
            Linear indices of basin outlets.
        xy : tuple of array_like of float, optional
            x and y coordinates of basin outlets.
        ids : 1D array of uint32, optional
            Basin IDs in the same order as `idxs`, by default None.
        **kwargs : dict
            Additional keyword arguments passed to `snap()` when outlet coordinates or
            indices are provided.

        Returns
        -------
        2D array of uint32
            Basin-ID raster, with zero as background.
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

    def subbasins(self, riv_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a subbasin map with unique IDs starting from 1.

        Parameters
        ----------
        riv_mask : 2D array of bool
            Boolean mask of river cells. For example, derive it from a minimum
            upstream-area or stream-order threshold.

        Returns
        -------
        subbas : 2D-array of int32
            Raster of unique subbasin IDs.
        idxs_out : 1D array of int
            Linear indices of subbasin outlet cells.
        """
        subbas, idxs_out = basins.subbasins(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            riv_mask=self._check_data(riv_mask, "riv_mask", optional=False),
            mv=self._mv,
        )
        return subbas.reshape(self.shape), idxs_out

    def subbasins_streamorder(
        self,
        strord: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        min_sto: int = -2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return subbasins defined by stream-order changes and their outlet indices.

        Subbasins are defined based on confluences where a lower order stream
        segment enters a higher order segment.

        Parameters
        ----------
        strord : 2D array of uint8, optional
            Stream-order map. Calculated from the flow directions if omitted.
        mask : 2D array of bool, optional
            Mask restricting valid stream cells.
        min_sto : int, optional
            Minimum stream order for subbasins. By default, uses two orders below the
            global maximum stream order.

        Returns
        -------
        subbas : 2D-array of int32
            Raster of unique IDs for subbasins meeting `min_sto`.
        idxs_out : 1D array of int
            Linear indices of subbasin outlet cells.
        """
        subbas, idxs_out = basins.subbasins_streamorder(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            strord=self._check_data(strord, "strord"),
            mask=self._check_data(mask, "mask", optional=True),
            min_sto=min_sto,
        )
        return subbas.reshape(self.shape), idxs_out

    def subbasins_pfafstetter(
        self,
        depth: int = 1,
        uparea: np.ndarray | None = None,
        upa_min: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return subbasins using the Pfafstetter coding system.

        Parameters
        ----------
        depth : int, optional
            Number of Pfafstetter coding levels, by default 1.
        uparea : 2D array of float, optional
            Raster of upstream area [km2]. Calculated on the fly if omitted.
        upa_min : float, optional
            Minimum upstream-area threshold for subbasins [km2], by default 0.0.

        Returns
        -------
        subbas: 2D array of int32
            Raster with Pfafstetter-coded subbasins.
        idxs_out : 1D array of int
            Linear indices of subbasin outlet cells.
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

    def subbasins_area(
        self, area_min: float, uparea: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return subbasins with a minimum contributing area of `area_min`.

        Moving upstream from basin outlets, a new subbasin starts at tributaries whose
        contributing area exceeds `area_min`. A new interbasin starts when its area
        exceeds the same threshold.

        Parameters
        ----------
        area_min : float
            Minimum subbasin area [km2].
        uparea : 2D array of float, optional
            Raster of upstream area [km2]. Calculated on the fly if omitted.

        Returns
        -------
        subbas: 2D array of int32
            Raster of unique subbasin IDs.
        idxs_out : 1D array of int
            Linear indices of subbasin outlet cells.
        """
        subbas, idxs_out = basins.subbasins_area(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            idxs_us_main=self.idxs_us_main,
            uparea=self._check_data(uparea, "uparea", unit="km2"),
            area_min=area_min,
        )
        return subbas.reshape(self.shape), idxs_out

    def basin_bounds(
        self, basins: np.ndarray | None = None, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the bounding boxes of basins.

        If `basins` is omitted, additional keyword arguments are passed to `basins()`
        to create the basin map.

        Parameters
        ----------
        basins : 2D array of uint32, optional
            Raster of basin IDs. Calculated on the fly if omitted.
        **kwargs : dict
            Additional keyword arguments passed to `basins()` when `basins` is omitted.

        Returns
        -------
        lbs : 1D array of int
            Basin IDs.
        bboxs : 2D array of float
            Bounding boxes with columns `[xmin, ymin, xmax, ymax]`.
        total_bbox : 1D array of float
            Bounding box enclosing all basins, `[xmin, ymin, xmax, ymax]`.
        """
        lbs, bboxs, total_bbox = regions.region_bounds(
            regions=self._check_data(basins, "basins", flatten=False, **kwargs),
            transform=self.transform,
        )
        return lbs, bboxs, total_bbox

    def basin_outlets(self, basins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return basin IDs and the linear index of each outlet cell.

        Parameters
        ----------
        basins : 2D array of int
            Raster of basin IDs. Each basin must be connected, and zero is reserved for
            background.

        Returns
        -------
        lbs : 1D array of int
            Unique basin IDs, sorted in ascending order.
        idxs_out : 1D array of int
            Linear index of the outlet cell for each basin in `lbs`.
        """
        lbs, idxs_out = regions.region_outlets(
            regions=self._check_data(basins, "basins"),
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
        )
        return lbs, idxs_out

    def interbasin_mask(
        self, region: np.ndarray, stream: np.ndarray | None = None
    ) -> np.ndarray:
        """Return a mask for the most downstream contiguous area within a region.

        If a stream enters and leaves the region, only the downstream portion within the
        region is included. If `stream` is provided, the mask is further restricted to
        cells that drain to a stream cell.

        Parameters
        ----------
        region : 2D array of bool
            Boolean mask of the region.
        stream : 2D array of bool, optional
            Boolean mask of stream cells.

        Returns
        -------
        mask : 2D array of bool
            True for cells in the selected area.
        """
        mask = basins.interbasin_mask(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            region=self._check_data(region, "region"),
            stream=self._check_data(stream, "stream", optional=True),
        )
        return mask.reshape(self.shape)

    ### ACCUMULATE ####

    def upstream_area(
        self, unit: Literal["m2", "ha", "km2", "cell"] = "cell"
    ) -> np.ndarray:
        """Return the upstream-area raster for the flow directions.

        Areas are returned in the units specified by `unit`.

        Parameters
        ----------
        unit : {'m2', 'ha', 'km2', 'cell'}
            Upstream-area units: square metres ('m2'), hectares ('ha'), square kilometres
            ('km2'), or cells ('cell'), by default 'cell'.

        Returns
        -------
        2D array
            Upstream area in `unit`; no-data cells are set to -9999.
        """
        if unit not in gis.AREA_FACTORS:
            fstr = '", "'.join(gis.AREA_FACTORS.keys())
            raise ValueError(f'Unknown unit: {unit}, select from "{fstr}".')
        area: np.ndarray
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
    def inflow_idxs(self, region: np.ndarray) -> np.ndarray:
        """Return linear indices of the most upstream cells within a region.

        Parameters
        ----------
        region : 2D array of bool
            True for cells inside the region.

        Returns
        -------
        1D array of int
            Linear indices of the most upstream cells in `region`.
        """
        return core.inflow_idxs(
            self.idxs_ds, self.idxs_seq, self._check_data(region, "region")
        )

    def outflow_idxs(self, region: np.ndarray) -> np.ndarray:
        """Return linear indices of the most downstream cells within a region.

        Parameters
        ----------
        region : 2D array of bool
            True for cells inside the region.

        Returns
        -------
        1D array of int
            Linear indices of the most downstream cells in `region`.
        """
        return core.outflow_idxs(
            self.idxs_ds, self.idxs_seq, self._check_data(region, "region")
        )

    def stream_distance(
        self, mask: np.ndarray | None = None, unit: Literal["m", "cell"] = "cell"
    ) -> np.ndarray:
        """Return the distance to the outlet or the next downstream True cell in `mask`.

        Parameters
        ----------
        mask : 2D-array of bool, optional
            True for stream cells.
        unit : {'m', 'cell'}, optional
            Distance unit, either metres ('m') or cells ('cell'), by default 'cell'.

        Returns
        -------
        2D array of float
            Distance to the next downstream True cell, or to the outlet, in `unit`.
        """
        if unit not in ["m", "cell"]:
            raise ValueError(f'Unknown unit: {unit}, select from "m", "cell"')
        stream_dist = streams.stream_distance(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            ncol=self.shape[1],
            mask=self._check_data(mask, "mask", optional=True),
            real_length=unit != "cell",
            transform=np.asarray(self.transform),
            latlon=self.latlon,
        )
        return stream_dist.reshape(self.shape)

    def vectorize(
        self,
        mask: np.ndarray | None = None,
        xs: np.ndarray | None = None,
        ys: np.ndarray | None = None,
        direction: Literal["up", "down"] = "down",
        **kwargs,
    ) -> list[dict]:
        """Return each selected flow path as a LineString feature.

        Parameters
        ----------
        mask : 2D array of bool, optional
            Mask selecting cells to include in the flow paths.
        xs, ys : 2D array of float, optional
            Rasters of cell-center x and y coordinates. Inferred from the affine
            transform if omitted.
        direction : {"up", "down"}
            Flow direction used to trace the paths, by default 'down'.
        **kwargs : dict
            Additional maps sampled at each feature's most downstream cell and included
            as feature properties.

        Returns
        -------
        feats : list of dict
            LineString features, suitable for `geopandas.GeoDataFrame.from_features`.
        """
        idxs = core.flwdir_tuples(
            self.idxs_ds if direction == "down" else self.idxs_us_main,
            mask=self._check_data(mask, "mask", optional=True),
            mv=self._mv,
        )
        return self.geofeatures(idxs, xs=xs, ys=ys, **kwargs)

    def streams(
        self,
        mask: np.ndarray | None = None,
        min_sto: int = 1,
        xs: np.ndarray | None = None,
        ys: np.ndarray | None = None,
        idxs_out: np.ndarray | None = None,
        max_len: int = 0,
        direction: Literal["up", "down"] = "up",
        **kwargs,
    ) -> list[dict]:
        """Return stream segments as LineString geographic features.

        A segment follows a flow path between confluences, or between outlet cells
        supplied through `idxs_out`. Stream cells are selected with either `mask` or
        `min_sto`; when `mask` is provided, `min_sto` is ignored.

        Additional keyword arguments provide maps sampled at the most downstream cell
        of each stream segment.

        Parameters
        ----------
        mask : 2D array of bool
            Mask selecting stream cells. If provided, `min_sto` is ignored.
        min_sto : int
            Minimum Strahler order recognized as a stream, by default 1. An existing
            stream-order map can be passed through the `strord` keyword argument.
        xs, ys : 2D array of float
            Rasters of cell-center x and y coordinates. Inferred from the affine
            transform if omitted.
        idxs_out : 1D array of int, optional
            Linear indices of segment-end cells. Segments follow the path between
            successive end cells in the direction specified by `direction`. By default,
            segments are defined by confluences.
        direction : {'up', 'down'}, optional
            Flow direction between segment-end cells. Used only when `idxs_out` is
            provided, by default 'up'.
        max_len: int, optional
            Maximum length of a single stream segment measured in cells.
            Longer segments are divided into shorter segments with lengths as close as
            possible to `max_len`, by default 0 (no maximum length).
        **kwargs : 2D array-like
            Additional maps sampled at each segment's most downstream cell and included
            as feature properties.

        Returns
        -------
        feats : list of dict
            LineString features, suitable for `geopandas.GeoDataFrame.from_features`.
        """
        if mask is not None:
            mask = self._check_data(mask, "mask")
        elif min_sto > 1:
            strord = self._check_data(
                cast(np.ndarray | None, kwargs.get("strord")), "strord"
            )
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

    def geofeatures(
        self,
        flowpaths: list[np.ndarray],
        xs: np.ndarray | None = None,
        ys: np.ndarray | None = None,
        **kwargs,
    ) -> list[dict]:
        """Return geographic features for flow paths represented by linear indices.

        Coordinates are calculated from the affine transform at cell centers unless
        (subgrid) x- and y-coordinate rasters are provided.

        Parameters
        ----------
        flowpaths : list of 1D arrays of int
            Flow paths described by linear indices.
        xs, ys : 2D array of float, optional
            Rasters of cell-center x and y coordinates. Inferred from the affine
            transform if omitted.
        **kwargs : 2D array-like
            Additional maps sampled at each feature's most downstream cell and included
            as feature properties, for example `strord=flw.stream_order()`.

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

    def upscale(
        self,
        scale_factor: int,
        method: Literal["ihu", "eam_plus", "eam", "dmm"] = "ihu",
        uparea: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[FlwdirRaster, np.ndarray]:
        """Upscale a flow-direction network to a lower resolution.

        Available methods are Iterative Hydrography Upscaling (IHU) [2]_,
        Effective Area Method (EAM) [3]_ and Double Maximum Method (DMM) [4]_.

        This method supports D8 and LDD flow-direction data only.

        .. [2] Eilander, D. et al (2021).
            A hydrography upscaling method for scale-invariant parametrization of distributed hydrological models.
            Hydrology and Earth System Sciences, 25(9), 5287–5313.
            https://doi.org/10.5194/hess-25-5287-2021

        .. [3] Yamazaki, D. et al (2008).
            An Improved Upscaling Method to Construct a Global River Map.
            Proceedings of the 4th Asia-Pacific Hydrology and Water Resources (APHW) Conference.

        .. [4] Olivera, F. et al (2002).
            Extracting low-resolution river networks from high-resolution digital elevation models.
            Water Resources Research, 38(11), 13-1-13–18. https://doi.org/10.1029/2001WR000726

        Parameters
        ----------
        scale_factor : int
            Number of high-resolution cells along each side of an upscaled cell.
        method : {'ihu', 'eam_plus', 'eam', 'dmm'}
            Upscaling method, by default 'ihu'.
        uparea : 2D array of float or int, optional
            Raster of upstream area. Calculated on the fly if not provided.
        **kwargs : dict
            Additional keyword arguments passed to the selected upscaling method. For
            IHU, these include `minlen_ratio`, `minupa_ratio`, `r_ratio`, `niter`,
            `opt_rivlen`, `min_error`, and `pit_out_of_cell`.

        Returns
        ------
        flw : FlwdirRaster
            Upscaled flow-direction raster.
        idxs_out : 2D array of int
            Linear indices of the high-resolution outlet cells corresponding to each
            upscaled cell.
        """
        if self.ftype not in ["d8", "ldd"]:
            raise ValueError(
                "The upscale method only works for D8 or LDD flow-direction data."
            )
        methods = ["ihu", "eam_plus", "com2", "com", "eam", "dmm"]
        if method not in methods:
            methodstr = "', '".join(methods)
            raise ValueError(f"Unknown method: {method}, select from: '{methodstr}'")
        if "com" in method.lower():
            method_new = {"com": "eam_plus", "com2": "ihu"}[method.lower()]
            warnings.warn(f"{method} renamed to {method_new}.", DeprecationWarning)
            method = method_new  # type: ignore[assignment]
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

    def upscale_error(self, other: FlwdirRaster, idxs_out: np.ndarray) -> np.ndarray:
        """Return an error map for the upscaled flow directions.

        A flow direction is valid when the first outlet pixel downstream of a cell's
        outlet is located in the cell indicated by that flow direction.

        The returned values are 1 for valid links, 0 for erroneous links, and 255 for
        cells with missing flow-direction data.

        Parameters
        ----------
        other : FlwdirRaster
            Upscaled flow-direction raster to check.
        idxs_out : 2D array of int
            Linear indices of the high-resolution outlet cells for `other`.

        Returns
        -------
        flwerr : 2D array of uint8 with `other.shape`
            Error map: 1 for valid links, 0 for erroneous links, and 255 for no-data.
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

    def ucat_outlets(
        self,
        cellsize: int,
        uparea: np.ndarray | None = None,
        method: Literal["eam_plus", "dmm"] = "eam_plus",
    ) -> np.ndarray:
        """Return linear indices of unit-catchment outlet pixels.

        The supported methods are `eam_plus` and `dmm`.

        Parameters
        ----------
        cellsize : int
            Unit-catchment width and height in high-resolution cells.
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

    def ucat_area(
        self, idxs_out: np.ndarray, unit: Literal["m2", "ha", "km2", "cell"] = "cell"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the high-resolution unit-catchment map and low-resolution cell areas.

        Parameters
        ----------
        idxs_out : 2D array of int
            Linear indices of unit-catchment outlet cells.
        unit : {'m2', 'ha', 'km2', 'cell'}, optional
            Area units, by default 'cell'.

        Returns
        -------
        ucat_map : 2D array of int with `self.shape`
            Unit-catchment ID for each high-resolution cell.
        ucat_area : 2D array of float with `idxs_out.shape`
            Area of each low-resolution cell, in `unit`.
        """
        if unit not in gis.AREA_FACTORS:
            fstr = '", "'.join(gis.AREA_FACTORS.keys())
            raise ValueError(f'Unknown unit: {unit}, select from "{fstr}".')
        area: np.ndarray
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
        self, idxs_out: np.ndarray, hand: np.ndarray, depths: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the high-resolution unit-catchment map and flood volumes by depth.

        Parameters
        ----------
        idxs_out : 1D or 2D array of int
            Linear indices of unit-catchment outlet cells.
        hand : 2D array of float
            Height Above Nearest Drain (HAND) raster [m]; see `hand()`.
        depths : 1D array of float, optional
            Flood depths [m] at which to calculate volume. By default, uses 0.5, 1.0,
            1.5, 2.0, and 2.5 m.

        Returns
        -------
        ucat_map : 2D array of int with `self.shape`
            Unit-catchment ID for each high-resolution cell.
        ucat_vol : array of float with shape `(depths.size, *idxs_out.shape)`
            Flood volume for each outlet at each depth [m3].
        """
        if depths is None:
            depths = np.arange(0.5, 3.0, 0.5, dtype=np.float32)
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
        idxs_out: np.ndarray | None,
        mask: np.ndarray | None = None,
        direction: Literal["up", "down"] = "up",
        unit: Literal["m", "cell"] = "cell",
    ) -> np.ndarray:
        """Returns the subgrid river length [m] based on unit catchment outlet locations.
        A cell's subgrid river is defined by the path starting at the unit
        catchment outlet pixel moving up- or downstream until it reaches the next
        outlet pixel. If moving upstream and a pixel has multiple upstream neighbors,
        the pixel with the largest upstream area is selected.

        Parameters
        ----------
        idxs_out : 2D array of int, optional
            Linear indices of unit catchment outlets.
            If None, the cell size (instead of subgrid length) will be used.
        mask : 2D array of bool with self.shape, optional
            True for valid pixels. can be used to mask out pixels of small rivers.
        direction : {"up", "down"}
            Flow direction in which river length is measured, by default 'up'.
        unit : {'m', 'cell'}, optional
            River-length unit, either metres ('m') or cells ('cell'), by default 'cell'.

        Returns
        -------
        rivlen : 2D array of float with idxs_out.shape
            subgrid river length [m]
        """
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
        idxs_out: np.ndarray | None,
        elevtn: np.ndarray,
        length: float = 1000,
        direction: Literal["both", "up", "down"] = "both",
        method: Literal["mean", "lstsq"] = "mean",
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the subgrid river slope [m/m] estimated at unit-catchment outlets.

        The slope is estimated from the elevation around the outlet pixel
        (`direction='both'`), or between the outlet pixel and the next downstream
        (`direction='down'`) or next upstream (`direction='up'`) outlet pixel.

        Parameters
        ----------
        idxs_out : 2D array of int
            Linear indices of unit catchment outlets
        elevtn : 2D array of float with self.shape
            Elevation raster, required to calculate slope.
        length : float, optional
            Subgrid river length [m] over which to calculate the slope, by default
            1000 m. Only used in combination with direction = 'both'
        direction : {"both", "up", "down"}
            Flow direction in which river slope is measured, by default 'both'.
        method : {'mean', 'lstsq'}, optional
            Estimate slope from the net elevation difference divided by length
            ('mean') or by least-squares regression ('lstsq'), by default 'mean'.
        mask : 2D array of bool with self.shape, optional
            True for valid pixels. Can be used to mask out pixels of small rivers.

        Returns
        -------
        rivslp : 2D array of float with idxs_out.shape
            subgrid river slope [m/m]
        """
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
        idxs_out: np.ndarray | None,
        data: np.ndarray,
        weights: np.ndarray | None = None,
        nodata: float = -9999.0,
        mask: np.ndarray | None = None,
        direction: Literal["up", "down"] = "up",
    ) -> np.ndarray:
        """Return the average value over the subgrid river at unit-catchment outlets.

        The subgrid river is defined by the path starting at the unit
        catchment outlet pixel moving up- or downstream until it reaches the next
        outlet pixel. If moving upstream and a pixel has multiple upstream neighbors,
        the pixel with the largest upstream area is selected.

        Parameters
        ----------
        idxs_out : 2D array of int
            Linear indices of unit-catchment outlets. If None, use each raster cell as
            an outlet.
        data : 2D array
            Values to average.
        weights : 2D array, optional
            Weights used for averaging. Equal weights are used if omitted.
        nodata : float, optional
            Missing data value for cells outside domain, by default -9999.0
        mask : 2D array of bool with self.shape, optional
            True for valid pixels. Can be used to exclude pixels outside the river.
        direction : {"up", "down"}
            Flow direction in which the segment is defined, by default 'up'.

        Returns
        -------
        rivavg : 2D array of float with idxs_out.shape
            Average value for each subgrid river segment.
        """
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
        idxs_out: np.ndarray | None,
        data: np.ndarray,
        weights: np.ndarray | None = None,
        nodata: float = -9999.0,
        mask: np.ndarray | None = None,
        direction: Literal["up", "down"] = "up",
    ) -> np.ndarray:
        """Return the median value over the subgrid river at unit-catchment outlets.

        The subgrid river is defined by the path starting at the unit
        catchment outlet pixel moving up- or downstream until it reaches the next
        outlet pixel. If moving upstream and a pixel has multiple upstream neighbors,
        the pixel with the largest upstream area is selected.

        Parameters
        ----------
        idxs_out : 2D array of int
            Linear indices of unit-catchment outlets. If None, use each raster cell as
            an outlet.
        data : 2D array
            Values for which to calculate the median.
        weights : 2D array, optional
            Deprecated. This argument is ignored and will be removed in a future version.
        nodata : float, optional
            Missing data value for cells outside domain, by default -9999.0
        mask : 2D array of bool with self.shape, optional
            True for valid pixels. Can be used to exclude pixels outside the river.
        direction : {"up", "down"}
            Flow direction in which the segment is defined, by default 'up'.

        Returns
        -------
        rivmed : 2D array of float with idxs_out.shape
            Median value for each subgrid river segment.
        """
        if direction not in ["up", "down"]:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        if idxs_out is None:
            idxs_out = np.arange(self.size, dtype=np.intp).reshape(self.shape)
        # raise deprecation warning if weights are provided
        if weights is not None:
            warnings.warn(
                "The 'weights' argument is deprecated and will be removed in a future version.",
                DeprecationWarning,
            )
        rivmed = subgrid.segment_median(
            idxs_out=idxs_out.ravel(),
            idxs_nxt=self.idxs_ds if direction == "down" else self.idxs_us_main,
            data=self._check_data(data, "data"),
            nodata=nodata,
            mask=self._check_data(mask, "mask", optional=True),
            mv=self._mv,
        )
        shape = idxs_out.shape
        return rivmed.reshape(shape)

    ### ELEVATION ###

    def dem_dig_d4(
        self,
        elevtn: np.ndarray,
        rivmsk: np.ndarray | None = None,
        nodata: float = -9999.0,
    ) -> np.ndarray:
        """Return elevation adjusted to satisfy D4 connectivity along river cells.

        Each river cell has an adjacent orthogonal cell with an elevation no greater
        than its own.

        Parameters
        ----------
        elevtn : 2D array of float
            elevation raster
        rivmsk : 2D array of bool, optional
            River-cell mask. If omitted, all valid cells are considered.
        nodata : float, optional
            No-data value in `elevtn`, by default -9999.0.

        Returns
        -------
        elv_out : 2D array of float
            Hydrologically adjusted elevation raster.
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

    def hand(self, drain: np.ndarray, elevtn: np.ndarray) -> np.ndarray:
        """Return the height above the nearest drain (HAND).

        HAND is the relative vertical distance (drop) to the nearest downstream river,
        calculated from drainage-normalized topography and flow paths.

        Nobre A D et al. (2016) HAND contour: a new proxy predictor of inundation extent
            Hydrol. Process. 30 320-33

        Parameters
        ----------
        drain : 2D array of bool
            Boolean mask of drainage cells.
        elevtn : 2D array of float
            Elevation raster.

        Returns
        -------
        2D array of float
            HAND raster, with no-data cells set to -9999.
        """
        hand = dem.height_above_nearest_drain(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            drain=self._check_data(drain, "drain"),
            elevtn=self._check_data(elevtn, "elevtn"),
        )
        return hand.reshape(self.shape)

    def floodplains(
        self,
        elevtn: np.ndarray,
        uparea: np.ndarray | None = None,
        upa_min: float = 1000,
        b: float = 0.3,
    ) -> np.ndarray:
        """Identify floodplain cells using an upstream-area-scaled HAND threshold.

        Cells with upstream area at least `upa_min` define the drainage network. For
        each such cell, the HAND threshold is its upstream area raised to `b`. Upstream
        cells are included when their elevation above the downstream drainage cell does
        not exceed that threshold.

        Nardi, F. et al (2019). GFPLAIN250m, a global high-resolution dataset of Earth's
        floodplains. Scientific Data, 6(1), 180309. https://doi.org/10.1038/sdata.2018.309

        Parameters
        ----------
        elevtn : 2D array of float
            Elevation raster [m].
        uparea : 2D array of float, optional
            Upstream-area raster [km2]. Calculated on the fly if omitted.
        b : float, optional
            Exponent in the area-scaling relationship, by default 0.3.
        upa_min : float, optional
            Minimum upstream-area threshold for drainage cells [km2], by default 1000.

        Returns
        -------
        2D array of int8
            Floodplain mask: 1 for floodplain cells, 0 for valid non-floodplain cells,
            and -1 for no-data cells.
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

    @overload
    def _check_data(
        self,
        data: None,
        name: str,
        optional: Literal[True] = ...,
        flatten: bool = ...,
        **kwargs,
    ) -> None:
        ...

    @overload
    def _check_data(
        self,
        data: np.ndarray | float | None,
        name: str,
        optional: Literal[False] = ...,
        flatten: bool = ...,
        **kwargs,
    ) -> np.ndarray:
        ...

    @overload
    def _check_data(
        self,
        data: np.ndarray | float | None,
        name: str,
        optional: bool,
        flatten: bool = ...,
        **kwargs,
    ) -> np.ndarray | None:
        ...

    def _check_data(
        self,
        data,
        name,
        optional=False,
        flatten=True,
        **kwargs,
    ):
        """check or calculate upstream area cells; return flattened array"""
        if data is None and optional:
            return None
        if data is None:
            if name == "uparea":
                data = self.upstream_area(**kwargs)
            elif name == "basins":
                data = self.basins(**kwargs)
            elif name == "strord":
                data = self.stream_order(**kwargs)
        return super()._check_data(data, name, optional, flatten=flatten)

    def _check_idxs_xy(  # type: ignore[override]
        self,
        idxs: np.ndarray | None = None,
        xy: tuple | None = None,
        streams: np.ndarray | None = None,
    ) -> np.ndarray:
        if (xy is not None and idxs is not None) or (xy is None and idxs is None):
            raise ValueError("Either idxs or xy should be provided.")
        elif xy is not None:
            idxs = self.index(*xy)
        return super()._check_idxs_xy(idxs, streams)

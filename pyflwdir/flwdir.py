"""Main module for flow direction parsing and analysis."""

import logging
import pickle
import pprint
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
from numba import njit

from . import (
    arithmetics,
    core,
    dem,
    rivers,
    streams,
)

if TYPE_CHECKING:
    import pandas as pd

# export
__all__ = ["Flwdir", "from_dataframe"]

# logging
logger = logging.getLogger(__name__)


def _ensure_supported_index_dtype(
    idxs: np.ndarray | None, name: str
) -> np.ndarray | None:
    if idxs is not None and idxs.dtype == np.uint64:
        uint64_mv = np.iinfo(np.uint64).max
        int64_max = np.iinfo(np.int64).max
        if np.any((idxs > int64_max) & (idxs != uint64_mv)):
            raise ValueError(
                f'"{name}" contains indices which cannot be represented as int64.'
            )
        idxs = idxs.astype(np.int64)
    return idxs


@njit(cache=True)
def get_loc_idx(idxs: np.ndarray, idxs_ds: np.ndarray) -> np.ndarray:
    """Get linear indices of downstream cells."""
    idx_map = {idx: i for i, idx in enumerate(idxs)}
    # return i if idx_ds not in idx_map, i.e. idx is a pit
    idxs_ds0 = np.empty(idxs.size, dtype=np.intp)
    for i, idx_ds in enumerate(idxs_ds):
        idxs_ds0[i] = i
        if idx_ds in idx_map:
            idxs_ds0[i] = idx_map[idx_ds]
    return idxs_ds0


def from_dataframe(df: "pd.DataFrame", ds_col: str = "idx_ds") -> "Flwdir":
    """Create a Flwdir object from a dataframe with flow direction data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing flow-direction links.
    ds_col : str, optional
        Name of the column containing downstream indices, by default "idx_ds".

    Returns
    -------
    Flwdir
        Flow-direction graph.
    """

    idxs_ds = df[ds_col].values
    idxs = df.index.values
    return Flwdir(idxs_ds=get_loc_idx(idxs=idxs, idxs_ds=idxs_ds))


class Flwdir:
    """Flow direction parsed to general actionable format."""

    def __init__(
        self,
        idxs_ds: np.ndarray,
        area: np.ndarray | None = None,
        idxs_pit: np.ndarray | None = None,
        idxs_outlet: np.ndarray | None = None,
        idxs_seq: np.ndarray | None = None,
        nnodes: int | None = None,
        cache: bool = True,
    ):
        """Initialize a flow-direction graph from downstream-node indices.

        Parameters
        ----------
        idxs_ds : 1D-array of int
            Linear index of the next downstream node for each node.
        area : 1D array, optional
            Node-area values, cached for use by accumulation methods.
        idxs_pit : 1D array of int, optional
            Linear indices of pit nodes. Calculated from `idxs_ds` if omitted.
        idxs_outlet : 1D array of int, optional
            Linear indices of basin outlet nodes.
        idxs_seq : 1D array of int, optional
            Valid node indices ordered from downstream to upstream.
        nnodes : int, optional
            Number of valid nodes. Calculated when needed if omitted.
        cache : bool, optional
            Whether to cache derived arrays, by default True.
        """
        # dimension
        self.size = idxs_ds.size
        if self.size <= 1:
            raise ValueError(f"Invalid FlwdirRaster: size {self.size}")
        # size for a 1D Flwdir; (nrow, ncol) for a FlwdirRaster
        self.shape: Any = self.size

        # data
        idxs_ds = cast(np.ndarray, _ensure_supported_index_dtype(idxs_ds, "idxs_ds"))
        idxs_pit = _ensure_supported_index_dtype(idxs_pit, "idxs_pit")
        idxs_outlet = _ensure_supported_index_dtype(idxs_outlet, "idxs_outlet")
        idxs_seq = _ensure_supported_index_dtype(idxs_seq, "idxs_seq")
        self._idxs_ds = idxs_ds
        self._pit = idxs_pit
        self.idxs_outlet = idxs_outlet
        self._seq = idxs_seq
        self._nnodes = nnodes
        # either -1 for signed integers or 4294967295 for uint32
        self._mv: Any = core._mv
        if idxs_ds.dtype == np.uint32:
            self._mv = np.uint32(self._mv)

        # set placeholders only used if cache if True
        self.cache = cache
        self._cached: dict = {}
        if area is not None:
            self._cached.update(area=area)

        # check validity
        if self.idxs_pit.size == 0:
            raise ValueError("Invalid FlwdirRaster: no pits found")

    ### REPRESENTATION ###

    def __str__(self) -> str:
        return pprint.pformat(self._dict)

    def __getitem__(self, idx):
        return self.idxs_ds[idx]

    ### PROPERTIES ###

    @property
    def _dict(self) -> dict:
        return {
            "nnodes": self.nnodes,
            "idxs_ds": self.idxs_ds,
            "idxs_seq": self._seq,
            "idxs_pit": self._pit,
        }

    @property
    def idxs_ds(self) -> np.ndarray:
        """Linear indices of downstream cell."""
        return self._idxs_ds

    @property
    def idxs_us_main(self) -> np.ndarray:
        """Linear indices of main upstream cell, i.e. the upstream cell with the
        largest contributing area."""
        if "idxs_us_main" in self._cached:
            idxs_us_main = self._cached["idxs_us_main"]
        else:
            idxs_us_main = self.main_upstream()
        return idxs_us_main

    @property
    def idxs_seq(self) -> np.ndarray:
        """Linear indices of valid cells ordered from down- to upstream."""
        if self._seq is None:
            self.order_cells(method="walk")
        return cast(np.ndarray, self._seq)

    @property
    def idxs_pit(self) -> np.ndarray:
        """Linear indices of pits/outlets."""
        if self._pit is None:
            self._pit = core.pit_indices(self.idxs_ds)
        return self._pit

    @property
    def nnodes(self) -> int:
        """Number of valid cells."""
        if self._nnodes is None:
            self._nnodes = int(np.sum(self.rank >= 0))
        return self._nnodes

    @property
    def rank(self) -> np.ndarray:
        """Cell Rank, i.e. distance to the outlet in no. of cells."""
        if "rank" in self._cached:
            rank = self._cached["rank"]
        else:
            rank = core.rank(self.idxs_ds, mv=self._mv)[0].reshape(self.shape)
            if self.cache:
                self._cached.update(rank=rank)
        return rank

    @property
    def isvalid(self) -> bool:
        """True if the flow direction map is valid."""
        self._cached.pop("rank", None)
        return bool(np.all(self.rank != -1))

    @property
    def mask(self) -> np.ndarray:
        """Boolean array of valid cells in flow direction raster."""
        return self.idxs_ds != self._mv

    @property
    def distnc(self) -> np.ndarray:
        """Per-node distance weights; one graph link has unit length."""
        if "distnc" in self._cached:
            distnc = self._cached["distnc"]
        else:
            distnc = np.ones_like(self.idxs_ds, dtype=np.float32)
        return distnc

    @property
    def area(self) -> np.ndarray:
        """Per-node area weights; each node has unit area by default."""
        if "area" in self._cached:
            area = self._cached["area"]
        else:
            area = np.ones_like(self.idxs_ds, dtype=np.float32)
        return area

    @property
    def n_upstream(self) -> np.ndarray:
        """Number of immediate upstream connections for each node."""
        return core.upstream_count(self.idxs_ds, mv=self._mv).reshape(self.shape)

    ### SET/MODIFY PROPERTIES ###

    def order_cells(
        self, method: Literal["sort", "walk", "dfs", "topo"] = "walk"
    ) -> None:
        """Order cells from down- to upstream.

        Parameters
        ----------
        method: {'walk', 'dfs', 'topo', 'sort'}, optional
            Method to order nodes. The default "walk" traces the nodes from down-
            to upstream breadth-first, holding the upstream cells of the whole
            network in memory in compressed sparse row layout. "dfs" traces them
            depth-first with the same index, which keeps each subbasin together
            in the sequence and may improve locality when the sequence is
            consumed. "topo" releases a node once all of its upstream nodes have
            been ordered, which needs a count per node instead of the upstream
            index.
            "sort" sorts the nodes on their rank, which can be slower for large
            arrays.

        Notes
        -----
        Every method returns the same cells, those that drain to a pit, in a
        sequence in which each cell other than a pit comes after the cell it
        drains into, which is what the flow network methods need: upstream
        area, basins, stream order and the like give the same result for each.
        The relative order of cells that do not drain into one another differs
        though, so labels given in sequence order (subbasins_streamorder) and
        the order of the features of streams change with the method, floating
        point accumulations can differ in the last bits, and dem_adjust and
        dem_dig_d4, which adjust the elevation one flow path at a time in
        sequence order, can give different adjustments.
        """
        if method == "sort":
            # slow for large arrays
            rnk, n = core.rank(self.idxs_ds, mv=self._mv)
            self._seq = np.argsort(rnk)[-n:].astype(self.idxs_ds.dtype)
        elif method == "walk":
            self._seq = core.idxs_seq(self.idxs_ds, self.idxs_pit, self._mv)
        elif method == "dfs":
            self._seq = core.idxs_seq_dfs(self.idxs_ds, self.idxs_pit, self._mv)
        elif method == "topo":
            self._seq = core.idxs_seq_topo(self.idxs_ds, self._mv)
        else:
            raise ValueError(
                f'Invalid method {method}, select from ["walk", "dfs", "topo", "sort"]'
            )
        self._nnodes = self._seq.size

    def main_upstream(self, uparea: np.ndarray | None = None) -> np.ndarray:
        """Return the main upstream node for each node.

        Parameters
        ----------
        uparea : 1D array of float, optional
            Upstream area used to select the main upstream node. Calculated from the
            graph if omitted.

        Returns
        -------
        1D array of int
            Linear indices of the selected upstream nodes.
        """
        idxs_us_main = core.main_upstream(
            idxs_ds=self.idxs_ds, uparea=self._check_data(uparea, "uparea"), mv=self._mv
        )
        if self.cache and uparea is None:
            self._cached.update(idxs_us_main=idxs_us_main)
        return idxs_us_main

    def add_pits(
        self, idxs: np.ndarray | None = None, streams: np.ndarray | None = None
    ) -> None:
        """Add pits to the flow direction graph.

        If `streams` is given, each pit is snapped to the first downstream True node.

        Parameters
        ----------
        idxs : array_like, optional
            Linear indices of pits.
        streams : 1D array of bool, optional
            One-dimensional boolean mask of stream nodes. When provided, pits in `idxs`
            are snapped to the first downstream True node.
        """
        idxs1 = self._check_idxs_xy(idxs, streams=streams)
        # add pits
        self.idxs_ds[idxs1] = idxs1
        self._pit = np.unique(np.concatenate([self.idxs_pit, idxs1]))
        # Reset traversal state and all values derived from the flow topology.
        self._seq = None
        self._nnodes = None
        for key in ("rank", "strord", "idxs_us_main", "distnc"):
            self._cached.pop(key, None)

    def repair_loops(self) -> None:
        """Repair loops by setting a pit at every cell which does not drain to a pit."""
        repair_idx = core.loop_indices(self.idxs_ds, mv=self._mv)
        if repair_idx.size > 0:
            # set pits for all loop indices !
            self.add_pits(repair_idx)

    ### IO ###

    def dump(self, fn: str) -> None:
        """Serialize the flow-direction graph to a file using pickle.

        Parameters
        ----------
        fn : str
            Output file path.
        """
        with open(fn, "wb") as handle:
            pickle.dump(self._dict, handle, protocol=-1)

    @staticmethod
    def load(fn: str) -> "Flwdir":
        """Load a serialized flow-direction graph from a file.

        Parameters
        ----------
        fn : str
            Input file path.

        Returns
        -------
        Flwdir
            Loaded flow-direction graph.
        """
        with open(fn, "rb") as handle:
            kwargs = pickle.load(handle)
        return Flwdir(**kwargs)

    ### LOCAL METHODS ###
    def path(
        self,
        idxs: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        max_length: float | None = None,
        direction: Literal["up", "down"] = "down",
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Trace paths upstream or downstream from starting nodes.

        Each path includes its starting node and ends at a pit, at a node where `mask`
        is True, or when the next link would exceed `max_length`.

        Parameters
        ----------
        idxs : 1D array of int
            Linear indices of starting nodes.
        mask : array-like of bool, optional
            True for path-end nodes. Use a 1D array for `Flwdir` and an array matching
            the raster shape for `FlwdirRaster`. Matching nodes are included.
        max_length : float, optional
            Maximum path length in links. Tracing stops before a link would exceed this
            distance.
        direction : {'up', 'down'}, optional
            Path direction, either downstream ('down', default) or upstream ('up').

        Returns
        -------
        paths : list of 1D arrays of int
            Linear indices for each traced path.
        distance : 1D array of float
            Number of links traversed from each starting node.
        """
        if direction not in ["up", "down"]:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        paths, dist = core.path(
            idxs0=idxs,
            idxs_nxt=self.idxs_ds if direction == "down" else self.idxs_us_main,
            mask=self._check_data(mask, "mask", optional=True),
            max_length=max_length,
            real_length=False,
            ncol=None,
            mv=self._mv,
        )
        return paths, dist

    ### GLOBAL ARITHMETICS ###

    def fillnodata(
        self,
        data: np.ndarray,
        nodata: float,
        direction: Literal["up", "down"] = "down",
        how: Literal["min", "max", "sum"] = "max",
    ) -> np.ndarray:
        """Fill no-data nodes with values from valid upstream or downstream neighbors.

        Parameters
        ----------
        data : array-like
            Values associated with graph cells. Use a 1D array for `Flwdir` and an
            array matching the raster shape for `FlwdirRaster`.
        nodata : int or float
            Missing-data value.
        direction : {'up', 'down'}, optional
            Direction in which to propagate values, downstream ('down', default) or
            upstream ('up').
        how : {'min', 'max', 'sum'}, optional
            Method to merge values at confluences. By default 'max'.
            Only used in combination with `direction = 'down'`.

        Returns
        -------
        array
            Data with no-data cells filled, with the same shape as `data`.
        """
        dflat = self._check_data(data, "data")
        if direction == "up":
            dout = core.fillnodata_upstream(self.idxs_ds, self.idxs_seq, dflat, nodata)
        elif direction == "down":
            dout = core.fillnodata_downstream(
                self.idxs_ds, self.idxs_seq, dflat, nodata, how=how
            )
        else:
            msg = 'Unknown flow direction: {direction}, select from ["up", "down"].'
            raise ValueError(msg)
        return dout.reshape(data.shape)

    def downstream(self, data: np.ndarray) -> np.ndarray:
        """Return the next downstream node's value for each node.

        Parameters
        ----------
        data : array-like
            Values associated with graph cells. Use a 1D array for `Flwdir` and an
            array matching the raster shape for `FlwdirRaster`.

        Returns
        -------
        array
            Values from the next downstream cells, with the same shape as `data`.
        """
        dflat = self._check_data(data, "data")
        data_out = dflat.copy()
        data_out[self.mask] = dflat[self.idxs_ds[self.mask]]
        return data_out.reshape(data.shape)

    def upstream_sum(self, data: np.ndarray, mv: float = -9999) -> np.ndarray:
        """Return the sum of values at each node's immediate upstream neighbors.

        Parameters
        ----------
        data : array-like
            Values associated with graph cells. Use a 1D array for `Flwdir` and an
            array matching the raster shape for `FlwdirRaster`.
        mv : int or float, optional
            Missing-data value, by default -9999.

        Returns
        -------
        array
            Sum of immediate upstream values at each cell, with the same shape as
            `data`.
        """
        data_out = arithmetics.upstream_sum(
            idxs_ds=self.idxs_ds,
            data=self._check_data(data, "data"),
            nodata=mv,
            mv=self._mv,
        )
        return data_out.reshape(data.shape)

    def moving_average(
        self,
        data: np.ndarray,
        n: int,
        weights: np.ndarray | None = None,
        restrict_strord: bool = False,
        strord: np.ndarray | None = None,
        nodata: float = -9999.0,
    ) -> np.ndarray:
        """Take the moving weighted average over the flow direction network

        Parameters
        ----------
        data : array-like
            Values associated with graph cells. Use a 1D array for `Flwdir` and an
            array matching the raster shape for `FlwdirRaster`.
        n : int
            number of up/downstream neighbors to include
        weights : array-like, optional
            Per-cell weights. Equal weights are used if omitted.
        restrict_strord: bool
            If True, limit the window to cells of same or smaller stream order.
        strord : array-like of int, optional
            Stream-order map used when `restrict_strord` is True.
        nodata : float, optional
            Nodata values which is ignored when calculating the average, by default -9999.0

        Returns
        -------
        array
            Averaged values, with the same shape as `data`.
        """
        data_out = arithmetics.moving_average(
            data=self._check_data(data, "data"),
            weights=self._check_data(weights, "weights", optional=True),
            n=n,
            idxs_ds=self.idxs_ds,
            idxs_us_main=self.idxs_us_main,
            strord=self._check_data(strord, "strord", optional=not restrict_strord),
            nodata=nodata,
            mv=self._mv,
        )
        return data_out.reshape(data.shape)

    def moving_median(
        self,
        data: np.ndarray,
        n: int,
        restrict_strord: bool = False,
        strord: np.ndarray | None = None,
        nodata: float = -9999.0,
    ) -> np.ndarray:
        """Take the moving median over the flow direction network

        Parameters
        ----------
        data : array-like
            Values associated with graph cells. Use a 1D array for `Flwdir` and an
            array matching the raster shape for `FlwdirRaster`.
        n : int
            number of up/downstream neighbors to include
        restrict_strord: bool
            If True, limit the window to cells of same or smaller stream order.
        strord : array-like of int, optional
            Stream-order map used when `restrict_strord` is True.
        nodata : float, optional
            Nodata values which is ignored when calculating the median, by default -9999.0

        Returns
        -------
        array
            Median values, with the same shape as `data`.
        """
        data_out = arithmetics.moving_median(
            data=self._check_data(data, "data"),
            n=n,
            idxs_ds=self.idxs_ds,
            idxs_us_main=self.idxs_us_main,
            strord=self._check_data(strord, "strord", optional=not restrict_strord),
            nodata=nodata,
            mv=self._mv,
        )
        return data_out.reshape(data.shape)

    ### STREAMS  ###

    def stream_order(
        self,
        type: Literal["strahler", "classic"] = "strahler",
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the Strahler (default) or classic stream-order map.

        In the *classic* bottom-up order, the main stem has order 1. Each tributary
        receives an order one greater than the stream it joins.

        In the *Strahler* top-down order, first-order streams are the most upstream
        tributaries, or headwater cells. When two streams of the same order merge, the
        downstream stream has an order one higher. When streams of different orders
        merge, the downstream stream takes the higher order.

        Parameters
        ----------
        type: {"strahler", "classic"}
            Stream-order type, by default 'strahler'.
        mask : array-like of bool, optional
            Mask of stream cells to consider. Use a 1D array for `Flwdir` and an array
            matching the raster shape for `FlwdirRaster`. Can restrict the calculation
            to streams above an upstream-area threshold or within a (sub)basin.

        Returns
        -------
        array of int
            Stream-order values for each cell, with the same shape as the flow-direction
            data.
        """
        mask = self._check_data(mask, "mask", optional=True)
        if type.lower() == "strahler":
            if mask is None and "strord" in self._cached:
                strord = self._cached["strord"]
            else:
                strord = streams.strahler_order(self.idxs_ds, self.idxs_seq, mask=mask)
                if self.cache and mask is None:
                    self._cached.update(strord=strord)
        elif type.lower() == "classic":
            strord = streams.stream_order(
                self.idxs_ds, self.idxs_seq, self.idxs_us_main, mask=mask, mv=self._mv
            )
        return strord.reshape(self.shape)

    def upstream_area(self) -> np.ndarray:
        """Returns the upstream area map based on the flow directions and set area.


        Returns
        -------
        nd array of float
            upstream area
        """
        uparea = streams.accuflux(
            idxs_ds=self.idxs_ds,
            seq=self.idxs_seq,
            data=self.area,
            nodata=-9999,
        )
        uparea[~self.mask] = -9999
        return uparea.reshape(self.shape)

    def accuflux(
        self,
        data: np.ndarray,
        nodata: float = -9999,
        direction: Literal["up", "down"] = "up",
    ) -> np.ndarray:
        """Return accumulated data values along the flow directions.

        Parameters
        ----------
        data : array-like
            Values associated with graph cells. Use a 1D array for `Flwdir` and an
            array matching the raster shape for `FlwdirRaster`.
        nodata : int or float
            Missing data value for cells outside domain
        direction : {'up', 'down'}, optional
            direction in which to accumulate data, by default upstream

        Returns
        -------
        array with `data.dtype`
            Accumulated values, with the same shape as `data`.
        """
        if direction == "up":
            accu = streams.accuflux(
                idxs_ds=self.idxs_ds,
                seq=self.idxs_seq,
                data=self._check_data(data, "data"),
                nodata=nodata,
            )
        elif direction == "down":
            accu = streams.accuflux_ds(
                idxs_ds=self.idxs_ds,
                seq=self.idxs_seq,
                data=self._check_data(data, "data"),
                nodata=nodata,
            )
        else:
            raise ValueError(
                'Unknown flow direction: {direction}, select from ["up", "down"].'
            )
        return accu.reshape(data.shape)

    def smooth_rivlen(
        self,
        rivlen: np.ndarray,
        min_rivlen: float,
        max_window: int = 10,
        nodata: float = -9999.0,
    ) -> np.ndarray:
        """Return smoothed river length, by taking the window average of river length.
        The window size is increased until the average exceeds the `min_rivlen` threshold
        or the `max_window` size is reached.

        Parameters
        ----------
        rivlen : array-like of float
            River length values.
        min_rivlen : float
            Minimum river length.
        max_window : int
            Maximum window size, by default 10.
        nodata : float, optional
            Missing-data value, by default -9999.0.

        Returns
        -------
        array of float
            Smoothed river-length values, with the same shape as `rivlen`.
        """
        rivlen_out = streams.smooth_rivlen(
            idxs_ds=self.idxs_ds,
            idxs_us_main=self.idxs_us_main,
            rivlen=self._check_data(rivlen, "rivlen"),
            min_rivlen=min_rivlen,
            max_window=max_window,
            nodata=nodata,
            mv=self._mv,
        )
        return rivlen_out.reshape(rivlen.shape)

    ### ELEVATION ###

    def dem_adjust(self, elevtn: np.ndarray) -> np.ndarray:
        """Returns the hydrologically adjusted elevation where each downstream cell
        has the same or lower elevation as the current cell.

        Parameters
        ----------
        elevtn : 2D array of float
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
            mv=self._mv,
        )
        return elevtn_out.reshape(elevtn.shape)

    ### RIVERS ###

    def classify_estuaries(
        self,
        elevtn: np.ndarray,
        rivwth: np.ndarray,
        rivdst: np.ndarray | None = None,
        min_convergence: float = 1e-2,
        max_elevtn: float = 0,
    ) -> np.ndarray:
        """Classify estuaries based on river-width convergence.

        Parameters
        ----------
        elevtn : 1D array of float
            Elevation values [m + reference elevation].
        rivwth : 1D array of float
            River widths [m].
        rivdst : 1D array of float, optional
            Distance-to-outlet values [m]. Uses the graph's `distnc` if omitted.
        max_elevtn : float, optional
            Maximum elevation for estuary outlets [m + reference elevation], by default 0.
        min_convergence : float, optional
            Minimum river-width convergence threshold [m/m], by default 1e-2.

        Returns
        -------
        np.ndarray of int8
            Estuary classification: 1 for estuary nodes, 2 at the upstream end of an
            estuary, and 0 elsewhere.
        """
        rivdst = self.distnc if rivdst is None else rivdst
        estuary = rivers.classify_estuary(
            self.idxs_ds,
            self.idxs_seq,
            self.idxs_pit,
            rivdst=self._check_data(rivdst, "rivdst"),
            rivwth=self._check_data(rivwth, "rivwth"),
            elevtn=self._check_data(elevtn, "elevtn"),
            min_convergence=min_convergence,
            max_elevtn=max_elevtn,
        )
        return estuary

    def river_depth(
        self,
        qbankfull: np.ndarray,
        rivwth: np.ndarray,
        zs: np.ndarray | None = None,
        rivdst: np.ndarray | None = None,
        rivslp: np.ndarray | None = None,
        manning: float | np.ndarray = 0.03,
        method: Literal["manning", "gvf"] = "manning",
        min_rivdph: float = 1,
        min_rivslp: float = 1e-5,
        **kwargs,
    ) -> np.ndarray:
        """Estimate river depth from Manning's equation or a gradually varied-flow solver.

        Both methods assume a rectangular river profile. The GVF method requires `zs`
        and `rivdst`. The Manning method requires `rivslp`, or both `zs` and `rivdst`
        from which to derive slope.

        Parameters
        ----------
        qbankfull : np.ndarray
            Bankfull discharge [m3/s].
        rivwth : np.ndarray
            Bankfull river width [m].
        zs : np.ndarray, optional
            Bankfull water-surface elevation [m + reference elevation]. Required for
            the GVF method and for deriving slope when `rivslp` is not provided.
        rivdst : np.ndarray, optional
            Distance-to-outlet values [m]. Required for the GVF method and for deriving
            slope when `rivslp` is not provided.
        rivslp : np.ndarray, optional
            River slope [m/m]. Required by the Manning method unless both `zs` and
            `rivdst` are supplied.
        manning : float, optional
            Manning roughness [s/m^(1/3)], by default 0.03.
        method : {'manning', 'gvf'}
            Method to estimate river depth, either 'manning' or 'gvf', by default
            'manning'.
        min_rivdph : float, optional
            Minimum river depth [m], by default 1.
        min_rivslp : float, optional
            Minimum river slope [m/m], by default 1e-5.
        **kwargs : dict
            Additional arguments passed to the GVF solver, including `eps` and
            `n_iter`.

        Returns
        -------
        rivdph: np.ndarray
            River-depth values [m].
        """
        methods = ["manning", "gvf"]
        if method not in methods:
            raise ValueError(f"Method unknown {method}, select from {methods}")
        # required arguments
        manning = self._check_data(manning, "manning")
        qbankfull = self._check_data(qbankfull, "qbankfull")
        rivwth = self._check_data(rivwth, "rivwth")
        # in case of manning either rivslp or zs&rivdst are optional
        _opt = method == "manning" and rivslp is not None
        rivslp = self._check_data(rivslp, "rivslp", optional=True)
        rivdst = self._check_data(rivdst, "rivdst", optional=_opt)
        zs = self._check_data(zs, "zs", optional=_opt)
        # get (initial) river slope from zs & rivdst
        if rivslp is None:
            if zs is None or rivdst is None:
                raise ValueError('"zs" and "rivdst" are required if "rivslp" is None.')
            dz = zs - self.downstream(zs)
            dx = rivdst - self.downstream(rivdst)
            rivslp = np.where(dx >= 1, dz / np.maximum(1, dx), -9999)
            rivslp = self.fillnodata(rivslp, nodata=-9999)
        rivslp = np.maximum(min_rivslp, rivslp)
        # get (initial) river depth based on manning's equation
        rivdph = ((manning * qbankfull) / (np.sqrt(rivslp) * rivwth)) ** (3 / 5)
        rivdph = np.maximum(min_rivdph, rivdph)
        rivdph[self.idxs_ds == self._mv] = -9999.0
        # update river depth based on contraint gradually varying flow solver
        if method == "gvf":
            if zs is None or rivdst is None:
                raise ValueError('"zs" and "rivdst" are required for the gvf method.')
            rivdph = rivers.rivdph_gvf(
                self.idxs_ds,
                self.idxs_seq,
                zs=zs,
                rivdph=rivdph,
                qbankfull=qbankfull,
                rivdst=rivdst,
                rivwth=rivwth,
                manning=manning,
                min_rivslp=min_rivslp,
                min_rivdph=min_rivdph,
                **kwargs,
            )
        return rivdph.reshape(self.shape)

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
        """check data shape and size; by default return flattened array"""
        if data is None and optional:
            return None
        if data is None:
            if name == "uparea":
                data = self.upstream_area(**kwargs)
            elif name == "strord":
                data = self.stream_order(**kwargs)
        data = np.atleast_1d(data)
        if flatten:
            if data.size == 1:
                data = np.full(self.size, data, dtype=data.dtype)
            elif data.size != self.size:
                raise ValueError(f'"{name}" size does not match.')
            return data.ravel()
        else:
            if data.size == 1:
                data = np.full(self.shape, data, dtype=data.dtype)
            elif data.shape != self.shape:
                raise ValueError(f'"{name}" shape does not match.')
            return data

    def _check_idxs_xy(
        self, idxs: np.ndarray | None = None, streams: np.ndarray | None = None
    ) -> np.ndarray:
        if idxs is None:
            raise ValueError('"idxs" should be provided.')
        idxs = np.atleast_1d(idxs).ravel()
        # snap to streams
        streams = self._check_data(streams, "streams", optional=True)
        if streams is not None:
            idxs = core.snap(
                idxs0=idxs, idxs_nxt=self.idxs_ds, mask=streams, mv=self._mv
            )[0]
        return idxs

# -*- coding: utf-8 -*-
""""""

import numpy as np
import pprint
import pickle
import logging
from numba import njit

from . import (
    arithmetics,
    core,
    dem,
    streams,
    rivers,
)

# export
__all__ = ["Flwdir", "from_dataframe"]

# logging
logger = logging.getLogger(__name__)


@njit(cache=True)
def get_loc_idx(idxs: np.ndarray, idxs_ds: np.ndarray) -> np.ndarray:
    """Get linear indices of downstream cells."""
    idx_map = {idx: i for i, idx in enumerate(idxs)}
    # return i if idx_ds not in idx_map, i.e. idx is a pit
    idxs_ds0 = np.empty_like(idxs, dtype=idxs.dtype)
    for i, idx_ds in enumerate(idxs_ds):
        idxs_ds0[i] = idx_map.get(idx_ds, i)
    return idxs_ds0


def from_dataframe(df: "pandas.DataFrame", ds_col="idx_ds") -> "Flwdir":
    """Create a Flwdir object from a dataframe with flow direction data.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe with flow directions
    ds_col : str, optional
        name of column with downstream indices, by default "idx_ds"

    Returns
    -------
    Flwdir
        flow direction object
    """

    idxs_ds = df[ds_col].values
    idxs = df.index.values
    return Flwdir(idxs_ds=get_loc_idx(idxs=idxs, idxs_ds=idxs_ds))


# def _get_idxs_ds_upstream(idxs: np.ndarray, idxs_up: np.ndarray) -> np.ndarray:
#     idxs_ds0 = np.arange(idxs.size, dtype=idxs.dtype)
#     for j, idx_up in enumerate(idxs_up):
#         for i, idx in enumerate(idxs):
#             if idx == idx_up:
#                 idxs_up0[j] = i
#     return idxs_ds0


class Flwdir(object):
    """Flow direction parsed to general actionable format."""

    def __init__(
        self,
        idxs_ds,
        area=None,
        idxs_pit=None,
        idxs_outlet=None,
        idxs_seq=None,
        nnodes=None,
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
            outlets exclude pits of inclomplete basins at the domain boundary
        idxs_seq : 2D array of int, optional
            linear indices of valid cells ordered from down- to upstream
        nnodes : integer
            number of valid cells
        """
        # dimension
        self.size = idxs_ds.size
        if self.size <= 1:
            raise ValueError(f"Invalid FlwdirRaster: size {self.size}")
        self.shape = self.size

        # data
        self._idxs_ds = idxs_ds
        self._pit = idxs_pit
        self.idxs_outlet = idxs_outlet
        self._seq = idxs_seq
        self._nnodes = nnodes
        # either -1 for int, 4294967295 for uint32, or 18446744073709551615 for uint64
        self._mv = core._mv
        if idxs_ds.dtype == np.uint32:
            self._mv = np.uint32(self._mv)
        if idxs_ds.dtype == np.uint64:
            self._mv = np.uint64(self._mv)

        # set placeholders only used if cache if True
        self.cache = cache
        self._cached = dict()
        if area is not None:
            self._cached.upate(area=area)

        # check validity
        if self.idxs_pit.size == 0:
            raise ValueError("Invalid FlwdirRaster: no pits found")

    ### REPRESENTATION ###

    def __str__(self):
        return pprint.pformat(self._dict)

    def __getitem__(self, idx):
        return self.idxs_ds[idx]

    ### PROPERTIES ###

    @property
    def _dict(self):
        return {
            "nnodes": self.nnodes,
            "idxs_ds": self.idxs_ds,
            "idxs_seq": self._seq,
            "idxs_pit": self._pit,
        }

    @property
    def idxs_ds(self):
        """Linear indices of downstream cell."""
        return self._idxs_ds

    @property
    def idxs_us_main(self):
        """Linear indices of main upstream cell, i.e. the upstream cell with the
        largest contributing area."""
        if "idxs_us_main" in self._cached:
            idxs_us_main = self._cached["idxs_us_main"]
        else:
            idxs_us_main = self.main_upstream()
        return idxs_us_main

    @property
    def idxs_seq(self):
        """Linear indices of valid cells ordered from down- to upstream."""
        if self._seq is None:
            self.order_cells(method="sort")
        return self._seq

    @property
    def idxs_pit(self):
        """Linear indices of pits/outlets."""
        if self._pit is None:
            self._pit = core.pit_indices(self.idxs_ds)
        return self._pit

    @property
    def nnodes(self):
        """Number of valid cells."""
        if self._nnodes is None:
            self._nnodes = int(np.sum(self.rank >= 0))
        return self._nnodes

    @property
    def rank(self):
        """Cell Rank, i.e. distance to the outlet in no. of cells."""
        if "rank" in self._cached:
            rank = self._cached["rank"]
        else:
            rank = core.rank(self.idxs_ds, mv=self._mv)[0].reshape(self.shape)
            if self.cache:
                self._cached.update(rank=rank)
        return rank

    @property
    def isvalid(self):
        """True if the flow direction map is valid."""
        self._cached.pop("rank", None)
        return np.all(self.rank != -1)

    @property
    def mask(self):
        """Boolean array of valid cells in flow direction raster."""
        return self.idxs_ds != self._mv

    @property
    def distnc(self):
        """Distance to outlet [m]"""
        if "distnc" in self._cached:
            distnc = self._cached["distnc"]
        else:
            distnc = np.ones_like(self.idxs_ds, dtype=np.float32)
        return distnc

    @property
    def area(self):
        """Cell area [m]"""
        if "area" in self._cached:
            area = self._cached["area"]
        else:
            area = np.ones_like(self.idxs_ds, dtype=np.float32)
        return area

    @property
    def n_upstream(self):
        """Number of upstream connection"""
        return core.upstream_count(self.idxs_ds, mv=self._mv).reshape(self.shape)

    ### SET/MODIFY PROPERTIES ###

    def order_cells(self, method="sort"):
        """Order cells from down- to upstream.

        Parameters
        ----------
        method: {'sort', 'walk'}, optional
            Method to order nodes, based on a "sorting" algorithm where nodes are
            sorted based on their rank (might be slow for large arrays) or "walking"
            algorithm where nodes are traced from down- to upstream (uses more memory)
        """
        if method == "sort":
            # slow for large arrays
            rnk, n = core.rank(self.idxs_ds, mv=self._mv)
            self._seq = np.argsort(rnk)[-n:].astype(self.idxs_ds.dtype)
        elif method == "walk":
            # faster for large arrays, but also takes lots of memory
            self._seq = core.idxs_seq(self.idxs_ds, self.idxs_pit, self._mv)
        else:
            raise ValueError(f'Invalid method {method}, select from ["walk", "sort"]')
        self._nnodes = self._seq.size

    def main_upstream(self, uparea=None):
        idxs_us_main = core.main_upstream(
            idxs_ds=self.idxs_ds, uparea=self._check_data(uparea, "uparea"), mv=self._mv
        )
        if self.cache:
            self._cached.update(idxs_us_main=idxs_us_main)
        return idxs_us_main

    def add_pits(self, idxs=None, streams=None):
        """Add pits the flow direction.
        If `streams` is given, the pits are snapped to the first downstream True node.

        Parameters
        ----------
        idxs : array_like, optional
            linear indices of pits, by default is None.
        streams : 1D array of bool, optional
            1D raster with cells flagged 'True' at stream nodes, only used
            in combination with idx, by default None.
        """
        idxs1 = self._check_idxs_xy(idxs, streams=streams)
        # add pits
        self.idxs_ds[idxs1] = idxs1
        self._pit = np.unique(np.concatenate([self.idxs_pit, idxs1]))
        # reset order, nnodes and upstream cell indices
        self._seq = None
        self._nnodes = None
        self._idxs_us_main = None

    def repair_loops(self):
        """Repair loops by setting a pit at every cell which does not drain to a pit."""
        repair_idx = core.loop_indices(self.idxs_ds, mv=self._mv)
        if repair_idx.size > 0:
            # set pits for all loop indices !
            self.add_pits(repair_idx)

    ### IO ###

    def dump(self, fn):
        """Serialize object to file using pickle library."""
        with open(fn, "wb") as handle:
            pickle.dump(self._dict, handle, protocol=-1)

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
        return Flwdir(**kwargs)

    ### LOCAL METHODS ###
    def path(
        self,
        idxs=None,
        mask=None,
        max_length=None,
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
        mask : 1D array of bool, optional
            True for path end nodes.
        max_length : float, optional
            maximum length of trace in number of nodes
        direction : {'up', 'down'}, optional
            direction of path, be default 'down', i.e. downstream

        Returns
        -------
        list of 1D-array of int
            linear indices of path
        1D-array of float
            distance along path between start and end cell
        """
        direction = str(direction).lower()
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

    def fillnodata(self, data, nodata, direction="down", how="max"):
        """Returns data where cells with nodata value have been filled
        with the nearest up- or downstream valid neighbor value.

        Parameters
        ----------
        data : 2D array
            values
        nodata: int, float
            missing data value
        direction : {'up', 'down'}, optional
            direction of path, be default 'down', i.e. downstream
        how: {'min', 'max', 'sum'}, optional.
            Method to merge values at confluences. By default 'max'.
            Only used in combination with `direction = 'down'`.

        Returns
        -------
        2D array
            filled data
        """
        direction = str(direction).lower()
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
        data_out = dflat.copy()
        data_out[self.mask] = dflat[self.idxs_ds[self.mask]]
        return data_out.reshape(data.shape)

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
            idxs_ds=self.idxs_ds,
            data=self._check_data(data, "data"),
            nodata=mv,
            mv=self._mv,
        )
        return data_out.reshape(data.shape)

    def moving_average(
        self, data, n, weights=None, restrict_strord=False, strord=None, nodata=-9999.0
    ):
        """Take the moving weighted average over the flow direction network

        Parameters
        ----------
        data : 2D array
            values
        n : int
            number of up/downstream neighbors to include
        weights : 2D array, optional
            weights, by default equal weights are assumed
        restrict_strord: bool
            If True, limit the window to cells of same or smaller stream order.
        strord : 2D array of int, optional
            Stream order map.
        nodata : float, optional
            Nodata values which is ignored when calculating the average, by default -9999.0

        Returns
        -------
        2D array
            averaged data
        """
        data_out = arithmetics.moving_average(
            data=self._check_data(data, "data"),
            weights=self._check_data(weights, "weights", optional=True),
            n=n,
            idxs_ds=self.idxs_ds,
            idxs_us_main=self.idxs_us_main,
            strord=self._check_data(strord, "strord", optional=~restrict_strord),
            nodata=nodata,
            mv=self._mv,
        )
        return data_out.reshape(data.shape)

    def moving_median(
        self, data, n, restrict_strord=False, strord=None, nodata=-9999.0
    ):
        """Take the moving median over the flow direction network

        Parameters
        ----------
        data : 2D array
            values
        n : int
            number of up/downstream neighbors to include
        restrict_strord: bool
            If True, limit the window to cells of same or smaller stream order.
        strord : 2D array of int, optional
            Stream order map.
        nodata : float, optional
            Nodata values which is ignored when calculating the median, by default -9999.0

        Returns
        -------
        2D array
            median data
        """
        data_out = arithmetics.moving_median(
            data=self._check_data(data, "data"),
            n=n,
            idxs_ds=self.idxs_ds,
            idxs_us_main=self.idxs_us_main,
            strord=self._check_data(strord, "strord", optional=~restrict_strord),
            nodata=nodata,
            mv=self._mv,
        )
        return data_out.reshape(data.shape)

    ### STREAMS  ###

    def stream_order(self, type="strahler", mask=None):
        """Returns the Strahler (default) or classic stream order map.

        In the *classic* "bottum up" stream order map, the main river stem has order 1.
        Each tributary is given a number one greater than that of the
        river or stream into which they discharge.

        In the *strahler* "top down" stream order map, rivers of the first order are
        the most upstream tributaries or head water cells. If two streams of the same
        order merge, the resulting stream has an order of one higher.
        If two rivers with different stream orders merge, the resulting stream is
        given the maximum of the two order.

        Parameters
        ----------
        type: {"strahler", "classic"}
            Stream order type. By default Strahler.
        mask: 2D array of boolean
            Mask of streams to consider. This can be used to compute the stream order
            for streams with a minimum upstream area or streams within a specific
            (sub)basin only.

        Returns
        -------
        2D array of int
            strahler order map
        """
        mask = self._check_data(mask, "mask", optional=True)
        if type.lower() == "strahler":
            if "strord" in self._cached:
                strord = self._cached["strord"]
            else:
                strord = streams.strahler_order(self.idxs_ds, self.idxs_seq, mask=mask)
                if self.cache:
                    self._cached.update(strord=strord)
        elif type.lower() == "classic":
            strord = streams.stream_order(
                self.idxs_ds, self.idxs_seq, self.idxs_us_main, mask=mask, mv=self._mv
            )
        return strord.reshape(self.shape)

    def upstream_area(self):
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

    def accuflux(self, data, nodata=-9999, direction="up"):
        """Return accumulated data values along the flow directions.

        Parameters
        ----------
        data : 2D array
            values
        nodata : int or float
            Missing data value for cells outside domain
        direction : {'up', 'down'}, optional
            direction in which to accumulate data, by default upstream

        Returns
        -------
        2D array with data.dtype
            accumulated values
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
        rivlen,
        min_rivlen,
        max_window=10,
        nodata=-9999.0,
    ):
        """Return smoothed river length, by taking the window average of river length.
        The window size is increased until the average exceeds the `min_rivlen` threshold
        or the `max_window` size is reached.

        Parameters
        ----------
        rivlen : 2D array of float
            River length values.
        min_rivlen : float
            Minimum river length.
        max_window : int
            maximum window size

        Returns
        -------
        2D array of float
            River length values.
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

    def dem_adjust(self, elevtn):
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
        self, elevtn, rivwth, rivdst=None, min_convergence=1e-2, max_elevtn=0
    ):
        """Classifies estuaries based on a minimum width convergence.

        Parameters
        ----------
        rivdst, rivwth, elevtn : np.ndarray
            Distance to river outlet [m], river width [m], elevation [m+REF]
        max_elevtn : float, optional
            Maximum elevation for estuary outlet, by default 0 m+REF
        min_convergence : float, optional
            River width convergence threshold, by default 1e-2 m/m

        Returns
        -------
        np.ndarray of int8
            Estuary classification: >= 1 where estuary; 2 at upstream end of estaury.
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
        qbankfull,
        rivwth,
        zs=None,
        rivdst=None,
        rivslp=None,
        manning=0.03,
        method="manning",
        min_rivdph=1,
        min_rivslp=1e-5,
        **kwargs,
    ):
        """Return an estimated river depth based on mannings equations or a gradually
        varying flow (gvf) solver a assuming a rectangular river profile.

        Parameters
        ----------
        qbankfull : np.ndarray
            bankfull discharge [m^3/s]
        rivwth : np.ndarray
            bankfull river width [m]
        zs : np.ndarray, optional
            bankfull water surface elevation profile [m+REF], required for gvf method
        rivdst : np.ndarray, optional
            distance to river outlet [m], required for gvf method
        rivslp : np.ndarray, optional
            river slope [m/m], required if `zs` or `rivdst` is not provided
        manning : float, optional
            manning roughness [s/m^{1/3}], by default 0.03
        method : {'manning', 'gvf'}
            Method to estimate river depth, by default 'manning'
        min_rivdph : int, optional
            Minimum river depth [m], by default 1
        min_rivslp : [type], optional
            Minimum river slope [m/m], by default 1e-5

        Returns
        -------
        rivdph: np.ndarray
            river depth [m]
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

    def _check_data(self, data, name, optional=False, flatten=True, **kwargs):
        """check data shape and size; by default return flattened array"""
        if data is None and optional:
            return
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

    def _check_idxs_xy(self, idxs, streams=None):
        idxs = np.atleast_1d(idxs).ravel()
        # snap to streams
        streams = self._check_data(streams, "streams", optional=True)
        if streams is not None:
            idxs = self.snap(idxs=idxs, mask=streams)[0]
        return idxs

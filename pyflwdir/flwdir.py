# -*- coding: utf-8 -*-
""""""

import numpy as np
import pprint
import pickle
import logging
from numba import njit

from pyflwdir import (
    arithmetics,
    core,
    streams,
    bathymetry,
)

# export
__all__ = ["Flwdir", "from_dataframe"]

# logging
logger = logging.getLogger(__name__)


@njit
def get_lin_indices(idxs, idxs_ds):
    idxs_ds0 = np.arange(idxs.size, dtype=idxs.dtype)
    for j, idx_ds in enumerate(idxs_ds):
        for i, idx in enumerate(idxs):
            if idx == idx_ds:
                idxs_ds0[j] = i
    return idxs_ds0


def from_dataframe(df, ds_col="idx_ds"):
    idxs_ds = df[ds_col].values
    idxs = df.index.values
    return Flwdir(idxs_ds=get_lin_indices(idxs=idxs, idxs_ds=idxs_ds))


class Flwdir(object):
    """Flow direction parsed to general actionable format."""

    def __init__(
        self,
        idxs_ds,
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
        # either -1 for int or 4294967295 for uint32
        self._mv = core._mv
        if idxs_ds.dtype == np.uint32:
            self._mv = np.uint32(self._mv)

        # set placeholders only used if cache if True
        self.cache = cache
        self._cached = dict()

        # check validity
        if self.idxs_pit.size == 0:
            raise ValueError("Invalid FlwdirRaster: no pits found")

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
    # TODO path, snap method

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
        data_out = arithmetics.moving_average(
            data=self._check_data(data, "data"),
            weights=self._check_data(weights, "weights", optional=True),
            n=n,
            idxs_ds=self.idxs_ds,
            idxs_us_main=self.idxs_us_main,
            nodata=nodata,
            mv=self._mv,
        )
        return data_out.reshape(data.shape)

    def moving_median(self, data, n, nodata=-9999.0):
        """Take the moving median over the flow direction network

        Parameters
        ----------
        data : 2D array
            values
        n : int
            number of up/downstream neighbors to include
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
            nodata=nodata,
            mv=self._mv,
        )
        return data_out.reshape(data.shape)

    ### STREAMS  ###

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
        if "strord" in self._cached:
            strord = self._cached["strord"]
        else:
            strord = streams.stream_order(self.idxs_ds, self.idxs_seq)
            if self.cache:
                self._cached.update(strord=strord)
        return strord.reshape(self.shape)

    ## bathymetry ###
    def depth_rect(
        self,
        q,
        n=None,
        s=None,
        w=None,
        z=None,
        d=None,
        h_pit=None,
        method="gvf",
        niter=3,
        force_monotonicity=True,
        **kwargs,
    ):
        """Returns river depth at each node.

        Parameters
        ----------

        """
        q = self._check_data(q, "discharge")
        n = self._check_data(n, "manning roughness", optional=method != "hdg")
        s = self._check_data(s, "slope", optional=method != "hdg")
        w = self._check_data(w, "width", optional=method != "hdg")
        d = self._check_data(d, "distance", optional=method != "gvf")
        z = self._check_data(z, "elevation", optional=not force_monotonicity)

        if method == "hdg":
            h_out = bathymetry.h_hdg(q, **kwargs)
        else:
            h_out = bathymetry.h_man(n, q, s, w)

        if h_pit is not None:
            npit = self.idxs_pit.size
            if h_pit.size > 1 and h_pit.size != npit:
                raise ValueError(
                    f"h_pit size does not match number of pits (n={npit})."
                )
            h_out[self.idxs_pit] = h_pit

        if force_monotonicity:
            for idx0 in self.idxs_seq:  # down- to upstream
                idx_ds = self.idxs_ds[idx0]
                if idx_ds == idx0:
                    continue
                zb_ds = z[idx_ds] - h_out[idx_ds]
                zb0 = z[idx0] - h_out[idx0]
                # make sure zb0 >= zb_ds
                h_out[idx0] = z[idx0] - max(zb_ds, zb0)

        if method == "gvf":
            n = n if isinstance(n, np.np.ndarray) else np.full_like(q, n)
            h_out1 = h_out.copy()
            for _ in range(niter):
                for idx0 in self.idxs_seq:  # down- to upstream
                    idx_ds = self.idxs_ds[idx0]
                    if idx_ds == idx0:
                        continue
                    h0, x0, x1 = h_out[idx_ds], d[idx_ds], d[idx0]
                    h1 = bathymetry.h_gvf(
                        h0, x0, x1, n[idx_ds], q[idx_ds], s[idx_ds], w[idx_ds], **kwargs
                    )
                    if force_monotonicity:
                        # make sure zb0 >= zb_ds
                        h1 = z[idx0] - max(z[idx_ds] - h0, z[idx0] - h1)
                    h_out1[idx0] = h1
                h_out = h_out1

        return h_out.reshape(self.shape)

    ### SHORTCUTS ###

    def _check_data(self, data, name, optional=False, **kwargs):
        """check or calculate upstream area cells; return flattened array"""
        if data is None and optional:
            return
        data = np.atleast_1d(data)
        if data.size == 1:
            data = np.full(self.size, data)
        elif data.size != self.size:
            raise ValueError(f'"{name}" size does not match.')
        return data.ravel()

    def _check_idxs_xy(self, idxs, streams=None):
        idxs = np.atleast_1d(idxs).ravel()
        # snap to streams
        streams = self._check_data(streams, "streams", optional=True)
        if streams is not None:
            idxs = self.snap(idxs=idxs, mask=streams)[0]
        return idxs

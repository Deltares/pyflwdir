# -*- coding: utf-8 -*-
"""Tests for the unitcatchments.py submodule"""

import pytest
import numpy as np

from pyflwdir import subgrid, core, streams

# test data
from test_core import test_data

parsed, flwdir = test_data[0]
idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in parsed]
ncol, shape = flwdir.shape[1], flwdir.shape
upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
idxs_us_main = core.main_upstream(idxs_ds, upa, mv=mv)
elv = rank

test = [("eam_plus", 5), ("", 1), ("dmm", 4)]


@pytest.mark.parametrize("method, cellsize", test)
def test_subgridch(method, cellsize):
    if cellsize == 1:
        idxs_out = np.arange(idxs_ds.size)
        idxs_out[idxs_ds == mv] = mv
    else:
        idxs_out, _ = subgrid.outlets(
            idxs_ds, upa, cellsize, shape, method=method, mv=mv
        )
    umap, uare = subgrid.ucat_area(idxs_out, idxs_ds, seq, ncol, dtype=np.int32, mv=mv)
    # upstream
    rivlen = subgrid.channel_length(idxs_out, idxs_us_main, ncol, latlon=True, mv=mv)
    rivslp = subgrid.channel_slope(
        idxs_out, idxs_ds, idxs_us_main, elv, ncol, length=1, latlon=True, mv=mv
    )
    rivwth = subgrid.channel_average(
        idxs_out, idxs_us_main, np.ones(elv.size), np.ones(elv.size), mv=mv
    )
    if cellsize == 1:
        assert np.all(uare[umap != 0] == cellsize)
        assert np.all(rivlen[upa == 1] == 0)  # headwater cells
        assert np.all(rivlen[upa > 1] >= 1)  # downstream cells
        assert np.all(np.isclose(rivslp[rivlen > 0], 1 / rivlen[rivlen > 0]))
    assert np.all(rivwth[idxs_out != mv] >= 0)  # downstream cells
    assert np.all(rivslp[idxs_out != mv] >= 0)
    assert np.all(rivlen[idxs_out != mv] >= 0)
    assert umap.max() - 1 == np.where(idxs_out != mv)[0][-1]
    assert np.all(uare[idxs_out != mv] >= 1)
    # downstream
    rivlen1 = subgrid.channel_length(idxs_out, idxs_ds, ncol, latlon=True, mv=mv)
    pits = idxs_ds[idxs_out[idxs_out != mv]] == idxs_out[idxs_out != mv]
    assert np.all(rivlen1[idxs_out != mv][pits] == 0)
    assert np.all(rivlen1[idxs_out != mv] >= 0)
    # mask
    rivlen2 = subgrid.channel_length(
        idxs_out, idxs_us_main, ncol, mask=None, latlon=True, mv=mv
    )
    rivlen3 = subgrid.channel_length(
        idxs_out, idxs_us_main, ncol, mask=upa >= 5, latlon=True, mv=mv
    )
    assert np.all(rivlen2 >= rivlen3)

    # TODO remove in v0.5
    wth = np.ones(rank.size)
    rivlen, rivslp, rivwth = subgrid.channel(
        idxs_out,
        idxs_ds,
        idxs_us_main,
        elv,
        wth,
        upa,
        ncol,
        len_min=1e6,
        latlon=True,
        mv=mv,
    )
    assert np.all(rivslp[idxs_out != mv] >= 0)
    assert np.all(rivlen[idxs_out != mv] >= 0)
    assert umap.max() - 1 == np.where(idxs_out != mv)[0][-1]
    assert np.all(uare[idxs_out != mv] >= 1)
    assert np.all(rivwth[idxs_out != mv] == 1)

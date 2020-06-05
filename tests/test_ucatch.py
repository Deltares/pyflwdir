# -*- coding: utf-8 -*-
"""Tests for the unitcatchments.py submodule"""

import pytest
import numpy as np

from pyflwdir import unitcatchments as ucat
from pyflwdir import core, streams

# test data
from test_core import test_data

parsed, flwdir = test_data[0]
idxs_ds, idxs_pit, seq, rank, mv = [p.copy() for p in parsed]
ncol, shape = flwdir.shape[1], flwdir.shape
upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
idxs_us_main = core.main_upstream(idxs_ds, upa, mv=mv)
elv = rank
wth = np.ones(rank.size)

test = [("eam", 5), ("none", 1), ("dmm", 4)]


@pytest.mark.parametrize("method, cellsize", test)
def test_ucatch(method, cellsize):
    if cellsize == 1:
        idxs_out = np.arange(idxs_ds.size)
        idxs_out[idxs_ds == mv] = mv
    else:
        idxs_out, _ = ucat.outlets(idxs_ds, upa, cellsize, shape, method=method, mv=mv)
    umap, uare = ucat.area(idxs_out, idxs_ds, seq, ncol, dtype=np.int32, mv=mv)
    # upstream
    rivlen, rivslp, rivwth = ucat.channel(
        idxs_out, idxs_us_main, elv, wth, upa, ncol, latlon=True, mv=mv
    )
    if cellsize == 1:
        assert np.all(uare[umap != 0] == cellsize)
        assert np.all(rivlen[upa == 1] == 0)  # headwater cells
        assert np.all(rivlen[upa > 1] >= 1)  # downstream cells
        # dz == 1 (elv == rank)
        assert np.all(np.isclose(rivslp[rivlen > 0], 1 / rivlen[rivlen > 0]))
    assert np.all(rivslp[idxs_out != mv] >= 0)
    assert np.all(rivlen[idxs_out != mv] >= 0)
    assert umap.max() - 1 == np.where(idxs_out != mv)[0][-1]
    assert np.all(uare[idxs_out != mv] >= 1)
    assert np.all(rivwth[idxs_out != mv] == 1)
    # downstream and no upa & wth
    rivlen1, rivslp1, rivwth1 = ucat.channel(
        idxs_out, idxs_ds, elv, None, None, ncol, latlon=True, mv=mv
    )
    pits = idxs_ds[idxs_out[idxs_out != mv]] == idxs_out[idxs_out != mv]
    assert np.all(rivlen1[idxs_out != mv][pits] == 0)
    assert np.all(rivslp1[idxs_out != mv] <= 0)
    assert np.all(rivlen1[idxs_out != mv] >= 0)
    assert np.all(rivwth1[idxs_out != mv] == -9999)

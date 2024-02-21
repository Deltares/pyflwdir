# -*- coding: utf-8 -*-
"""Tests for the unitcatchments.py submodule"""

import numpy as np
import pytest

from pyflwdir import core, streams, subgrid


@pytest.mark.parametrize("method, cellsize", [("eam_plus", 5), ("", 1), ("dmm", 4)])
def test_subgridch(method, cellsize, test_data0, flwdir0):
    idxs_ds, _, seq, rank, mv = [p.copy() for p in test_data0]
    ncol, shape = flwdir0.shape[1], flwdir0.shape
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    idxs_us_main = core.main_upstream(idxs_ds, upa, mv=mv)
    elv = rank

    if cellsize == 1:
        idxs_out = np.arange(idxs_ds.size)
        idxs_out[idxs_ds == mv] = mv
    else:
        idxs_out, _ = subgrid.outlets(
            idxs_ds, upa, cellsize, shape, method=method, mv=mv
        )
    umap, uare = subgrid.ucat_area(
        idxs_out, idxs_ds, seq, area=np.ones(idxs_ds.size, dtype=np.int32), mv=mv
    )
    # upstream
    rivlen = subgrid.segment_length(idxs_out, idxs_us_main, distnc=rank.ravel(), mv=mv)
    rivslp = subgrid.fixed_length_slope(
        idxs_out, idxs_ds, idxs_us_main, elv, rank.ravel(), mv=mv
    )
    rivwth = subgrid.segment_average(
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
    rivlen1 = subgrid.segment_length(idxs_out, idxs_ds, distnc=rank.ravel(), mv=mv)
    pits = idxs_ds[idxs_out[idxs_out != mv]] == idxs_out[idxs_out != mv]
    assert np.all(rivlen1[idxs_out != mv][pits] == 0)
    assert np.all(rivlen1[idxs_out != mv] >= 0)
    # mask
    rivlen2 = subgrid.segment_length(idxs_out, idxs_us_main, distnc=rank.ravel(), mv=mv)
    rivlen3 = subgrid.segment_length(
        idxs_out, idxs_us_main, distnc=rank.ravel(), mask=upa >= 5, mv=mv
    )
    assert np.all(rivlen2 >= rivlen3)

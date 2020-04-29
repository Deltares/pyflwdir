# -*- coding: utf-8 -*-
"""Tests for the pyflwdir.core.py submodule."""

import pytest
import numpy as np

from pyflwdir import core, core_d8, streams

_mv = core._mv

# test data
# slice from real data
flwdir0 = np.loadtxt("flwdir.asc", dtype=np.uint8)
idxs_ds0, idxs_pit0, _ = core_d8.from_array(flwdir0)
rank0, n0 = core.rank(idxs_ds0)
seq0 = np.argsort(rank0)[-n0:]
parsed0 = (idxs_ds0, idxs_pit0, seq0, rank0)
# random data (likely to contain loops!)
flwdir1 = np.random.choice(core_d8._ds.flatten(), (15, 10))
idxs_ds1, idxs_pit1, _ = core_d8.from_array(flwdir1)
rank1, n1 = core.rank(idxs_ds1)
seq1 = np.argsort(rank1)[-n1:]
parsed1 = (idxs_ds1, idxs_pit1, seq1, rank1)
# combined
test_data = [(parsed0, flwdir0), (parsed1, flwdir1)]


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_downstream(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank = [p.copy() for p in parsed]
    n, ncol = np.sum(idxs_ds != _mv), flwdir.shape[1]
    # rank
    assert np.sum(rank == 0) == idxs_pit.size
    if np.any(rank > 0):
        idxs_mask = np.where(rank > 0)[0]  # valid and no pits
        assert np.all(rank[idxs_mask] == rank[idxs_ds[idxs_mask]] + 1)
    # pit indices
    idxs_pit1 = np.sort(core.pit_indices(idxs_ds))
    assert np.all(idxs_pit1 == np.sort(idxs_pit))
    # loop indices
    idxs_loop = core.loop_indices(idxs_ds)
    assert seq.size == n - idxs_loop.size
    # local upstream indices
    if np.any(rank >= 2):
        rmax = np.max(rank)
        idxs = np.where(rank == rmax)[0]
        # path
        paths, dists = core.path(idxs, idxs_ds)
        assert np.all([p.size for p in paths] == rmax + 1)
        assert np.all(dists == rmax)
        # snap
        idxs1, dists1 = core.snap(idxs, idxs_ds, ncol, real_length=True)
        assert np.all([idxs_ds[idx] == idx for idx in idxs1])
        assert np.all(dists1 >= rmax)
        idxs2, dists2 = core.snap(idxs, idxs_ds, real_length=False, max_length=2)
        assert np.all(dists2 == 2)
        assert np.all(rank[idxs2] == rmax - 2)
        idxs2, dists2 = core.snap(idxs, idxs_ds, mask=rank <= rmax - 2)
        assert np.all(dists2 == 2)
        assert np.all(rank[idxs2] == rmax - 2)
        # window
        idx0 = np.where(rank == 2)[0][0]
        path = core._trace(idx0, idxs_ds)[0]
        wdw = core._window(idx0, 2, idxs_ds, idxs_ds)
        assert np.all(path == wdw[2:]) and np.all(path[::-1] == wdw[:-2])


@pytest.mark.parametrize("parsed, flwdir", test_data)
def test_upstream(parsed, flwdir):
    idxs_ds, idxs_pit, seq, rank = [p.copy() for p in parsed]
    idxs_ds[rank == -1] = _mv
    n, ncol = np.sum(idxs_ds != _mv), flwdir.shape[1]
    upa = streams.upstream_area(idxs_ds, seq, ncol, dtype=np.int32)
    # count
    n_up = core.upstream_count(idxs_ds)
    assert np.sum(n_up[n_up != -9]) == n - idxs_pit.size
    # upstream matrix
    idxs_us = core.upstream_matrix(idxs_ds)
    assert np.sum(idxs_us != _mv) == seq.size - idxs_pit.size
    # ordered
    seq2 = core.idxs_seq(idxs_ds, idxs_pit, flwdir.shape)
    assert np.all(np.diff(rank.flat[seq2]) >= 0)
    # headwater
    idxs_headwater = core.headwater_indices(idxs_ds)
    assert np.all(n_up[idxs_headwater] == 0)
    if np.any(n_up > 0):
        # local upstream indices
        idx0 = np.where(upa == np.max(upa))[0][0]
        idxs_us0 = np.sort(core._upstream_d8_idx(idx0, idxs_ds, flwdir.shape))
        idxs_us1 = np.sort(idxs_us[idx0, : n_up[idx0]])
        assert np.all(idxs_us1 == idxs_us0)
        # main upstream
        idxs_us_main = core.main_upstream(idxs_ds, upa)
        assert np.any(idxs_us0 == idxs_us_main[idx0])
        idxs = np.where(idxs_us_main != _mv)[0]
        assert np.all(idxs_ds[idxs_us_main[idxs]] == idxs)
        assert idxs.size == np.sum(n_up[upa > 0] >= 1)
        # tributary
        idxs_us_trib = core.main_tributary(idxs_ds, idxs_us_main, upa)
        idxs = np.where(idxs_us_trib != _mv)[0]
        assert idxs.size == np.sum(n_up[upa > 0] > 1)
        if idxs.size > 0:
            assert np.all(idxs_ds[idxs_us_main[idxs]] == idxs)
        # window
        path = core._trace(idx0, idxs_us_main, ncol)[0]
        wdw = core._window(idx0, 1, idxs_us_main, idxs_us_main)
        assert np.all(path[:2] == wdw[1:]) and np.all(path[:2][::-1] == wdw[:-1])
        # tributaries
        idxs_trib = core._tributaries(idx0, idxs_us_main, idxs_us_trib, upa)
        assert np.all([np.any(idx == idxs_us_trib[path]) for idx in idxs_trib])
        if idxs_trib.size > 1:
            idxs_trib1 = core._tributaries(idx0, idxs_us_main, idxs_us_trib, upa, n=1)
            assert np.max(upa[idxs_trib]) == upa[idxs_trib1]
